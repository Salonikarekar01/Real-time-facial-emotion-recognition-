# Real-Time Facial Emotion Recognition

A real-time facial emotion recognition system built using deep learning and computer vision.  
The project analyzes human facial expressions through live webcam input or uploaded images and predicts emotional states using trained CNN models.

---

## 🚀 Project Overview

This project focuses on building a robust real-time emotion recognition pipeline using convolutional neural networks (CNNs) and transfer learning techniques. Multiple models were trained, evaluated, and compared to study their performance, accuracy, and efficiency in real-world scenarios.

The system supports:
- Live webcam-based emotion detection
- Manual image upload for emotion analysis
- Comparative evaluation of different deep learning architectures

---

## 🧠 Models & Techniques Used

- Fine-tuned **MobileNetV2** (trained on CK+ and RAF-DB datasets)
- Custom CNN for emotion classification
- Transfer learning & fine-tuning
- Image preprocessing and face detection using OpenCV

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Computer Vision:** OpenCV  
- **Datasets:** CK+, RAF-DB, FER-2013  
- **Interface:** Streamlit  
- **Tools:** NumPy, Matplotlib  

---

## 📊 Features

- Real-time emotion recognition via webcam
- Image-based emotion prediction
- Comparison of multiple CNN architectures
- Model evaluation using accuracy and performance metrics
- User-friendly Streamlit interface

---

## 📁 Project Structure

├── models/
├── preprocessing/
├── app.py
├── train.py
├── evaluate.py
└── requirements.txt

## 🧰 Setup

1. Clone the repo:
   git clone https://github.com/Salonikarekar01/Real-time-facial-emotion-recognition-.git

2. Create virtual environment:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. Run the app:
   streamlit run app/app.py

### 📦 Datasets
- CK+ — facial expression dataset
- RAF-DB — real-world annotated facial dataset
- FER dataset for extended testing
 
## ⚠️ Notes

- Datasets are **not included** due to size constraints.
- This project is developed for **academic and learning purposes**.

---

## 📌 Future Improvements

- Improve accuracy using advanced architectures
- Deploy as a web service
- Add emotion tracking over time

---

## 👩‍💻 Author

**Saloni Mangesh Karekar**  
Artificial Intelligence & Data Science Student  

📫 Email: salonikarekar01@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/saloni-karekar/
