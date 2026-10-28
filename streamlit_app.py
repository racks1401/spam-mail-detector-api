import time
import threading
import requests
import streamlit as st

st.set_page_config(page_title="Spam Mail Detection", page_icon="📧")

BASE_URL = "https://spam-mail-detector-kzo4.onrender.com"
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


def wake_api(box, max_wait=120):
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
    return result["ok"]


def _call_api(message, out):
    try:
        r = requests.post(PREDICT_URL, json={"message": message}, timeout=60)
        out["data"] = r.json()
    except Exception as e:
        out["error"] = str(e)


def start_predict():
    st.session_state.predicting = True
    st.session_state.result = None


st.title("📧 Spam Mail Detection")
st.write("Machine Learning Spam Mail Detection Project (MCA & IoT).")

# Server status (green dot when connected)
status = st.empty()
if "api_ready" not in st.session_state:
    st.session_state.api_ready = wake_api(status)
if st.session_state.api_ready:
    status.markdown("🟢 **Server connected**")
else:
    status.markdown("🔴 **Server not connected.** Please refresh the page.")

if "predicting" not in st.session_state:
    st.session_state.predicting = False
if "result" not in st.session_state:
    st.session_state.result = None

st.text_area("Enter a message:", key="message", height=300)

st.button(
    "Predict",
    on_click=start_predict,
    disabled=st.session_state.predicting or not st.session_state.api_ready,
)

# Runs only right after the Predict click, button stays disabled meanwhile
if st.session_state.predicting:
    message = st.session_state.message.strip()
    if not message:
        st.session_state.result = {"warning": "Please enter a message!"}
    else:
        bar = st.progress(0, text="Predicting, please wait...")
        out = {}
        t = threading.Thread(target=_call_api, args=(message, out), daemon=True)
        t.start()
        pct = 0
        while t.is_alive():
            pct = min(pct + 3, 95)
            bar.progress(pct, text="Predicting, please wait...")
            time.sleep(0.1)
        bar.progress(100, text="Done")
        time.sleep(0.2)
        bar.empty()
        st.session_state.result = out
    st.session_state.predicting = False
    st.rerun()

# Show the result
res = st.session_state.result
if res:
    if "warning" in res:
        st.warning(res["warning"])
    elif "error" in res:
        st.error(f"API Error: {res['error']}")
    else:
        data = res["data"]
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