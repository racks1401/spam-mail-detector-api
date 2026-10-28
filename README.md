# 📧 Spam Mail Detector

A machine learning project that classifies a message as **Spam** or **Ham** (not spam). The model is served through a Flask REST API on Render, and a Streamlit web app gives it a simple interface.

**Live demo:** [YOUR_STREAMLIT_LINK](YOUR_STREAMLIT_LINK)

> The API runs on Render's free tier, so it goes to sleep when nobody uses it. The first load can take up to 50 seconds. The app shows a live timer while the server wakes up.

---

## Screenshots

<p align="center">
  <img src="/images/ham_image.png" alt="Spam Mail Detector - Ham result" width="48%">
  <img src="/images/spam_image.png" alt="Spam Mail Detector - Spam result" width="48%">
</p>

---

## Features

- Classifies any message as Spam or Ham
- Shows a confidence score with the prediction
- Streamlit UI with a server status indicator and a boot timer for the free-tier cold start
- Predict button locks while a request is running, with a progress bar
- REST API that can be used from any client, not just the Streamlit app

## Tech Stack

| Part | Tools |
|------|-------|
| Language | Python |
| ML | scikit-learn, joblib |
| Backend | Flask, Gunicorn |
| Frontend | Streamlit |
| Hosting | Render (API), Streamlit Community Cloud (UI) |

## How it works

1. The user types a message in the Streamlit app.
2. The app sends it to the Flask API (`POST /predict`).
3. The API converts the text with the saved vectorizer and runs the trained model.
4. The prediction and confidence score go back to the app and are shown on screen.

## API

**Health check**

```
GET /health
```

```json
{ "message": "Spam Detection API is running!" }
```

**Predict**

```
POST /predict
Content-Type: application/json
```

Request:

```json
{ "message": "Congratulations! You won a free prize, click here to claim." }
```

Response:

```json
{
  "input_message": "Congratulations! You won a free prize, click here to claim.",
  "prediction": "Spam Mail",
  "confidence": 91.66
}
```

An empty message returns `400` with an error message.

## Project Structure

```
├── app.py               # Flask API (deployed on Render)
├── streamlit_app.py     # Streamlit UI (deployed on Streamlit Cloud)
├── spam_model.pkl       # Trained model
├── vectorizer.pkl       # Saved text vectorizer
├── requirements.txt
└── Procfile             # Start command for Render
```

## Run Locally

```bash
git clone https://github.com/racks1401/spam-mail-detector-api.git
cd spam-mail-detector-api
pip install -r requirements.txt
```

Start the API:

```bash
python app.py
```

Start the Streamlit app in another terminal:

```bash
streamlit run streamlit_app.py
```

To test against your local API, change `BASE_URL` at the top of `streamlit_app.py` to `http://localhost:5000`.

## Author

**Ravi Kumar Prabudh**
MCA (AI & IoT), NIT Patna

[GitHub](https://github.com/racks1401) · [LinkedIn](https://www.linkedin.com/in/ravi-kumar-prabudh/)