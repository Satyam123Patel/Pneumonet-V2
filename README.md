# Pneumonia Detection Web Application (3-Class Model)

This is a complete web application designed to detect pneumonia from chest X-ray images. It uses a deep learning classification model trained on a 3-class dataset to differentiate between healthy lungs, infected lungs (pneumonia), and non-chest X-ray images (such as regular photos).

## Features
* **User Authentication**: Secure user registration, login, and session management.
* **Image Upload Dashboard**: Batch upload chest X-ray images for analysis.
* **3-Class Classification**:
  * `NORMAL`: Healthy chest X-ray.
  * `PNEUMONIA`: Chest X-ray showing signs of infection.
  * `NOT_XRAY`: Regular color photos/drawings that are not medical X-ray scans.
* **Confidence Visualizations**: Displays confidence percentages for each of the three classes.
* **Prediction History**: Saves all user upload histories and predictions securely in a local database.

---

## Technical Stack
* **Deep Learning Model**: EfficientNetB0 (Transfer Learning with TensorFlow/Keras)
* **Web Framework**: Flask
* **Database**: SQLite (SQLAlchemy ORM)
* **Image Preprocessing**: OpenCV

---

## Setup Instructions

### 1. Prerequisites
Make sure you have Python (version 3.10 to 3.12 recommended) installed on your system.

### 2. Clone the Repository & Install Dependencies
Clone this repository to your local machine and install the required Python libraries using:
```bash
pip install -r requirements.txt
```

### 3. Model Training & Dataset Preparation
The raw dataset contains `NORMAL` and `PNEUMONIA` images.
1. Download the Kaggle Chest X-Ray dataset.
2. Open **[project.ipynb](project.ipynb)** and run the cells:
   * Specify the path to your raw dataset in Cell 4: `ORIGINAL_DATA_DIR = r"C:\path\to\downloaded\chest_xray"`
   * Run the dataset preparation cells to partition a validation split and generate the synthetic `NOT_XRAY` class data.
   * Train the model (takes about 40 minutes on CPU; saves to `checkpoints_3class/best_model_final_3class.keras`).

---

## Running the Web Application

To start the local Flask server, run:
```bash
python app.py
```

Once started, open your web browser and navigate to:
**`http://localhost:5000`**

Register a new account or log in to start uploading images for prediction.
