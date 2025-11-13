# app.py
import streamlit as st
import tempfile
import time
import json
import os

from google import genai

# replace with your hardcoded key or use environment var (recommended)
API_KEY = ""
client = genai.Client(api_key=API_KEY)

st.title("Extract Sort Code & Account Number (Gemini Files API)")

uploaded_file = st.file_uploader("Upload image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

def upload_to_gemini(local_path):
    """
    Upload file to Gemini Files API using the SDK.
    Returns the uploaded file object (with .name or .id etc.)
    Note: method name may be client.files.upload or client.files.create depending on SDK; 
    the common pattern is client.files.upload(file=path) which returns a File resource.
    """
    # The google-genai SDK exposes a files.upload/ create in many examples.
    # Try client.files.upload(file=...) first (common example).
    try:
        file_obj = client.files.upload(file=local_path)
    except Exception as e:
        # fallback to client.files.create if upload() not present
        try:
            # create usually needs 'file' param too
            file_obj = client.files.create(file=local_path)
        except Exception as e2:
            raise RuntimeError(f"Failed to upload file via SDK: {e} | fallback: {e2}")
    return file_obj

def wait_for_file_ready(file_obj, timeout_s=60, poll_interval=1.0):
    """
    Poll until uploaded file state shows ready/processed. The SDK File object may have `.state` or `.name`.
    """
    start = time.time()
    name = getattr(file_obj, "name", None) or getattr(file_obj, "id", None) or str(file_obj)
    while True:
        # try to get freshest metadata (client.files.get)
        try:
            # some SDKs expose client.files.get(name=file_obj.name) or client.files.get(name=file_obj.id)
            latest = client.files.get(name=file_obj.name)
            # If state is present and indicates READY / PROCESSED, break
            state = getattr(latest, "state", None)
            if state:
                state_name = getattr(state, "name", None)
                if state_name and state_name.upper() in ("READY", "PROCESSED", "ACTIVE"):
                    return latest
            # in many setups, file is ready quickly – break if we don't have explicit state
            # but continue polling until timeout
            file_obj = latest
        except Exception:
            # best-effort: sometimes get() not available or file already okay
            pass

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for file {name} to be ready.")
        time.sleep(poll_interval)

if uploaded_file:
    st.image(uploaded_file, use_column_width=True)
    if st.button("Extract"):
        with st.spinner("Uploading file to Gemini Files API..."):
            # write uploaded bytes to a temporary file because SDK expects a path/file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            try:
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp.close()
                # Upload file to Gemini Files API
                try:
                    uploaded = upload_to_gemini(tmp.name)
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    raise

                st.info("Uploaded file. Waiting for processing...")
                try:
                    ready_file = wait_for_file_ready(uploaded, timeout_s=60, poll_interval=1.0)
                except Exception as e:
                    st.warning(f"File may not be fully processed but continuing: {e}")
                    ready_file = uploaded

                # Now call generate_content referencing the uploaded file resource
                # Many SDK examples accept the File object itself in contents.
                # We'll pass the file object followed by the prompt string.
                prompt = (
                    "Extract the sort code and account number from the referenced image. "
                    "Respond ONLY with JSON exactly like "
                    "{\"sort_code\":\"...\",\"account_number\":\"...\"} and nothing else."
                )

                # contents should be a flat list where file_obj is one element and prompt is another
                contents = [ready_file, prompt]

                with st.spinner("Calling Gemini generate_content..."):
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=contents
                    )

                raw = response.text.strip()
                try:
                    parsed = json.loads(raw)
                    st.success("Extraction result:")
                    st.json(parsed)
                except json.JSONDecodeError:
                    st.error("Gemini did not return strict JSON. Raw output shown below:")
                    st.code(raw)

            finally:
                # remove temp file
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
