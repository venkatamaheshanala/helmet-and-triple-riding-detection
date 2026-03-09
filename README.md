# hel_triple_ride_detection
# 🚀 YOLOv8-Based Helmet & Triple Riding Detection Web App

This project is a **real-time web-based object detection tool** built using [Streamlit](https://streamlit.io/) and [YOLOv8](https://docs.ultralytics.com/). It detects **helmet violations** and **triple riding on motorcycles**, focusing on road safety automation. Users can upload **images or videos**, and the app will display **only the top-confidence detection** per frame.

---

## 📌 Live Demo  

🚀 Try it here: [hel-triple-ride-detection Web App](https://hel-triple-ride-detection-1.onrender.com/)

---

## 🎯 Motivation  

Motorcycle accidents are a major concern, especially due to violations like **not wearing helmets** and **riding with more than two passengers**. Manual surveillance is:  
- ⏱️ Time-consuming  
- 👀 Error-prone  
- 🚔 Limited by available enforcement personnel  

To overcome this, an **AI-powered real-time detection system** is proposed to help traffic authorities monitor and enforce rules more effectively, ultimately reducing accidents and saving lives.

---

## 🧠 Model Training Overview  

- **Dataset**: 6,050 images annotated into 3 classes:  
  - ✅ Safe riding  
  - ❌ No helmet  
  - 👥 Triple riding (overloading)  

- **Preprocessing**: Resizing (640×640), normalization, and augmentations (rotation, flips, brightness adjustments, Mosaic, MixUp).  

- **Architecture**:  
  - **Backbone**: CSPDarknet + C2f modules  
  - **Neck**: PANet for multi-scale feature fusion  
  - **Head**: Anchor-free detection of helmets, no-helmets, and triple riding  

- **Training Setup**:  
  - Epochs: 100  
  - Batch size: 16  
  - Optimizer: AdamW with cosine annealing LR scheduler  
  - Loss: BCE + CIoU + Distribution Focal Loss  
  - Confidence threshold: 0.3  

- **Performance**:  
  - Precision: **95.1% (mAP@50)**  
  - Strong generalization across lighting, traffic density, and camera angles  

---

---

## 📊 Results  

- Overall **Precision**: 0.895  
- **Recall**: 0.863  
- **mAP@0.5**: 0.919  
- Best performance: **Triple riding detection** (Precision: 0.929, Recall: 0.935, mAP@0.5: 0.964)  
- Low misclassification rates with reliable confusion matrix trends  

---

## 🌐 Features of the Web App  

- 📤 Upload `.jpg`, `.png`, `.mp4`  
- 🏆 Displays only **highest-confidence detection per frame**  
- 🎚️ Adjustable confidence threshold  
- 🎨 Custom Streamlit UI theme  
- ⚡ Fast YOLOv8 inference with `best3.pt` model  

---

## ⚡ Setup & Run Locally  

```bash
# Clone the repository
git clone https://github.com/yourusername/hel_triple_ride_detection.git
cd hel_triple_ride_detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
