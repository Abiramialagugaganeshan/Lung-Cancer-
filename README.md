# 🫁 Lung Cancer Prediction with Grad-CAM  

This project is designed to detect **lung cancer from CT scan images** using **deep learning** and provide **visual explanations** through **Grad-CAM**. A simple **Flask-based web application** allows users to upload CT scan images, get predictions (Benign, Malignant, or Normal), and visualize the decision-making process with heatmaps.  

---

## 📂 Dataset  
The model is trained on the **IQ-OTHNCCD Lung Cancer Dataset**.  

- **Classes**:  
  - Benign  
  - Malignant  
  - Normal  

- **Preprocessing Steps**:  
  - Images resized to `224x224`  
  - Normalization of pixel values between `0–1`   

---

## 🔄 Workflow  
1. **Data Preprocessing** – Image resizing, normalization, dataset splitting  
2. **Model Development** – VGG16 fine-tuning with custom classification layers  
3. **Training & Evaluation** – Metrics: Accuracy, Precision, Recall, F1-score  
4. **Explainability** – Grad-CAM heatmaps highlight critical regions  
5. **Deployment** – Flask app for real-time predictions and visualizations  

---

## 🚀 Features  
- ✅ VGG16-based **deep learning model**  
- ✅ **Grad-CAM visualization** for interpretability  
- ✅ **Flask web app** for user interaction  
- ✅ **Personalized recommendations** with predictions  

---

## 🖥️ Application Workflow  
1. Upload a CT scan image via the web app  
2. System preprocesses and predicts the scan category  
3. Grad-CAM generates a **heatmap overlay**  
4. User sees the **prediction + medical recommendation**  

---
---

## 📽️ Demo Video  

<video width="600" controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*(Place your `demo.mp4` in the repo root along with README.md for GitHub to render it.)*  

---

## ⚙️ Installation & Usage  

### Clone the repository  
```bash
git clone https://github.com/Abiramialaguganeshan/lung-cancer-.git
cd lung-cancer-
