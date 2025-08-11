This project is a Deepfake Audio Detection System built using a hybrid forensic framework combining:

CNN on Mel-Spectrograms for deep spectral feature extraction.
Random Forest on MFCCs for interpretable acoustic analysis.
The app allows users to:

Upload an audio file (.wav / .flac)
View the prediction: Bonafide or Spoofed
See explainability outputs such as Grad-CAM heatmaps or SHAP plots (if enabled)

Dataset used: ASVspoof2019 Logical Access.

🚀 Features
Real-time audio deepfake detection
CNN model (Keras) trained on Mel-Spectrograms
Random Forest model (joblib) trained on MFCCs
Explainability tools:

Grad-CAM for CNN visual focus
SHAP for RF feature importance
Streamlit web interface