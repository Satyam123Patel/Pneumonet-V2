---
title: Pneumonet V2
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Pneumonet-V2: Pneumonia Detection Web Application (3-Class Model)
Pneumonet-V2 is a deep learning web application built using Flask and TensorFlow. It allows users to upload chest X-ray images to detect the presence of pneumonia. The model distinguishes between healthy lungs, lungs infected with pneumonia, and non-X-ray images (such as regular color photos) to prevent incorrect inputs.
Live Demo URL: `https://<your-deployed-app-url>.com` (Update this once deployed)
---
## 🚀 Features
* **User Authentication**: Secure user registration, login, and session management.
* **3-Class Classification**:
  * `NORMAL`: Healthy chest X-ray scan.
  * `PNEUMONIA`: Chest X-ray showing signs of bacterial or viral infection.
  * `NOT_XRAY`: Safety filter to reject regular color photos or drawings.
* **Confidence Bar Chart**: Displays percentage probabilities for all three classes.
* **History Dashboard**: Saves all previous predictions securely in a local database for users to view.
---
## 🛠️ Technical Stack
* **Deep Learning**: EfficientNetB0 (Transfer Learning in TensorFlow/Keras)
* **Backend**: Flask
* **Database**: SQLite & SQLAlchemy ORM
* **Image Preprocessing**: OpenCV & Pillow
---
## 💻 Local Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/Satyam123Patel/Pneumonet-V2.git
cd Pneumonet-V2
2. Install Dependencies
Ensure you have Python 3.10 to 3.12 installed, then run:

bash


pip install -r requirements.txt
3. Run the App
Start the local server:

bash


python app.py
Open http://127.0.0.1:5000 in your browser.

(Note: The pre-trained model best_model_final_3class.keras is included in the project root directory, so you can run the app immediately without retraining!)

📈 Model Training (Optional)
If you want to train the model from scratch on your own machine:

Download the Kaggle Chest X-Ray dataset.
Open project.ipynb or run train_model.py.
Set the directory path ORIGINAL_DATA_DIR to your downloaded dataset.
Execute the training script to generate a new model checkpoint.
