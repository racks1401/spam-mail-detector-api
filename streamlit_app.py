import time
import threading
import requests
import streamlit as st

st.set_page_config(page_title="Spam Mail Detection", page_icon="📧")

BASE_URL = "https://spam-mail-detector-kzo4.onrender.com"
HEALTH_URL = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict"


def _ping(result, deadline):
    while time.time() < deadline:
        try:
            if requests.get(BASE_URL, timeout=10).status_code < 500:
                result["ok"] = True
                return
        except requests.RequestException:
            pass
        time.sleep(1)

def wake_api(max_wait=120):
    box = st.empty()
    start = time.time()
    result = {"ok": False}
    t = threading.Thread(target=_ping, args=(result, start + max_wait), daemon=True)
    t.start()

    while t.is_alive():
        elapsed = int(time.time() - start)
        box.info(
            f"⏳ Server is starting up... {elapsed}s "
            "(free hosting, the first load can take up to 50 seconds)"
        )
        time.sleep(1)

    elapsed = int(time.time() - start)
    if result["ok"]:
        box.success(f"✅ Server is ready ({elapsed}s)")
        return True
    box.error("Server did not start. Please refresh the page and try again.")
    return False


st.title("📧 Spam Mail Detection")
st.write("Machine Learning Spam Mail Detection Project (MCA & IoT).")

if "api_ready" not in st.session_state:
    st.session_state.api_ready = wake_api()

message = st.text_area("Enter a message:")

if st.button("Predict", disabled=not st.session_state.api_ready):
    if not message.strip():
        st.warning("Please enter a message!")
    else:
        with st.spinner("Analyzing message..."):
            try:
                r = requests.post(PREDICT_URL, json={"message": message}, timeout=60)
                data = r.json()
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        prediction = data.get("prediction")
        if prediction == "Spam Mail":
            st.error("🚨 Prediction: Spam Mail")
        elif prediction:
            st.success(f"✔ Prediction: {prediction}")
        else:
            st.error(data.get("error", "Unknown error"))

        confidence = data.get("confidence")
        if confidence is not None:
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")