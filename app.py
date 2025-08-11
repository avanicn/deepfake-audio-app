import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import tempfile
import pandas as pd

# === Voting Function ===
def vote_model(rf_result, cnn_result):
    return rf_result if rf_result == cnn_result else "Spoof"

# === Load models and features ===
rf_model = joblib.load("best_rf_top7_model.joblib")
top_indices = joblib.load("top7_feature_indices.joblib")
cnn_model = load_model("spoof_audio_detector_models_1000trainingnow.keras")

# === Logging ===
def log_prediction(filename, rf_result, cnn_result, confidence):
    log_path = "predictions_log.csv"
    new_entry = {
        "filename": filename,
        "rf_result": rf_result,
        "cnn_result": cnn_result,
        "cnn_confidence": round(confidence, 4)
    }
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
    df.to_csv(log_path, index=False)

# === Feature Extraction ===
def extract_full_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.hstack([np.mean(mfcc, axis=1), np.mean(chroma, axis=1), np.mean(zcr)]).reshape(1, -1)

def predict_rf(full_features):
    selected = full_features[0][top_indices].reshape(1, -1)
    prob = rf_model.predict_proba(selected)[0]
    pred = np.argmax(prob)
    label = "Bonafide" if pred == 1 else "Spoof"
    return label, round(float(prob[pred]), 4)

# === CNN Predictions ===
def generate_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    plt.axis('off')
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(buf.name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return buf.name

def predict_cnn(image_path):
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img_input = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    pred = cnn_model.predict(img_input, verbose=0)[0][0]
    label = "Bonafide" if pred < 0.5 else "Spoof"
    confidence = 1 - pred if label == "Bonafide" else pred
    return label, round(float(confidence), 4)

# === Streamlit UI ===
st.markdown("<h1 style='text-align: center; color: navy;'> Deepfake Audio Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Hybrid Ensemble using CNN + Random Forest</h4>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("📁 Upload Audio File")
uploaded_file = st.sidebar.file_uploader("Choose a .flac or .wav file", type=["flac", "wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as tmp_audio:
        tmp_audio.write(uploaded_file.read())
        tmp_audio_path = tmp_audio.name

    st.audio(uploaded_file, format="audio/flac")
    full_features = extract_full_features(tmp_audio_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Random Forest (MFCCs)")
        if full_features.shape[1] > max(top_indices):
            rf_result, rf_confidence = predict_rf(full_features)
            st.write(f"**Prediction:** {rf_result}")
            st.write(f"**Confidence:** {rf_confidence:.2f}")
        else:
            st.error("Feature extraction failed or insufficient features.")

    with col2:
        st.subheader(" CNN (Spectrogram)")
        mel_img_path = generate_mel_spectrogram(tmp_audio_path)
        cnn_result, cnn_confidence = predict_cnn(mel_img_path)
        st.write(f"**Prediction:** {cnn_result}")
        st.write(f"**Confidence:** {cnn_confidence:.2f}")

    final_vote = vote_model(rf_result, cnn_result)
    st.markdown("---")
    st.subheader(" Final Ensemble Decision")
    st.markdown(f"<h3 style='color: green;'>Majority Vote Result: <u>{final_vote}</u></h3>", unsafe_allow_html=True)

    log_prediction(uploaded_file.name, rf_result, cnn_result, cnn_confidence)

    os.remove(mel_img_path)
    os.remove(tmp_audio_path)
