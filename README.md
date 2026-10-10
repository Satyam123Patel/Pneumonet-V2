---
title: Pneumonet V2
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Pneumonet-V2: Pneumonia Detection Web Application (3-Class Model)

Pneumonet-V2 is a deep learning web application built using Flask and TensorFlow. It allows users to upload chest X-ray images to detect the presence of pneumonia. The model distinguishes between healthy lungs, lungs infected with pneumonia, and non-X-ray images to prevent incorrect inputs.

---

## Features
* **Guest Preview Mode**: New visitors can run up to 3 instant predictions without an account.
* **User Authentication**: Secure user registration, login, and session management.
* **3-Class Classification**:
  * NORMAL: Healthy chest X-ray scan.
  * PNEUMONIA: Chest X-ray showing signs of bacterial or viral infection.
  * NOT_XRAY: Safety filter to reject non-medical photos.
* **Confidence Bar Chart**: Displays percentage probabilities for all three classes.
* **History Dashboard**: Saves all previous predictions in a database for registered users.

---

## Technical Stack
* **Deep Learning**: EfficientNetB0 (Transfer Learning in TensorFlow/Keras)
* **Backend**: Flask
* **Database**: SQLite and SQLAlchemy ORM
* **Image Preprocessing**: OpenCV and Pillow
* **Deployment**: Docker on Hugging Face Spaces (Port 7860)

---

## Local Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Satyam123Patel/Pneumonet-V2.git
cd Pneumonet-V2
```

### 2. Install Dependencies
Ensure you have Python 3.10 to 3.12 installed:
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## Model Training (Optional)
If you wish to inspect or retrain the model:
1. Open `project.ipynb` in Jupyter Notebook or Google Colab.
2. Download the Kaggle Chest X-Ray dataset.
3. Update the data paths and run all cells.
