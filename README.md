# Lung-Cancer-Prediction
🫁 Lung Cancer Prediction with Grad-CAM

This project focuses on early detection of lung cancer using deep learning and explainable AI techniques. A Flask-based web application allows users to upload CT scan images, predict whether the scan shows Benign, Malignant, or Normal conditions, and visualize the results with Grad-CAM heatmaps for interpretability.

📂 Dataset

The model is trained on the IQ-OTHNCCD Lung Cancer Dataset
.

Classes: Benign, Malignant, Normal

Preprocessing:

Images resized to 224x224

Normalized pixel values between 0–1

Data split into 80% training and 20% testing

🔄 Workflow

Data Preprocessing – Resize, normalize, and split dataset

Model Development – Fine-tuned VGG16 with custom layers

Training & Evaluation – Accuracy, Precision, Recall, F1-score used as metrics

Explainability – Applied Grad-CAM to highlight important regions in CT scans

Deployment – Flask app for predictions & heatmap visualization

🚀 Features

✅ Deep Learning Model (VGG16-based) – Accurate classification

✅ Grad-CAM Integration – Visual interpretability for model decisions

✅ Flask Web App – Upload CT scans & view predictions instantly

✅ Personalized Recommendations – Actionable medical guidance for each prediction

🖥️ Application Workflow

Upload a lung CT scan image

The system preprocesses and predicts the condition

Grad-CAM generates a heatmap overlay on the scan

The app displays the prediction + recommendation
