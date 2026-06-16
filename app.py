"""
Complete Pneumonia Detection Web Application
With User Authentication, Prediction History, and Database
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pneumonia_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
IMG_SIZE = 224
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Additional prediction details
    normal_prob = db.Column(db.Float, default=0.0)
    not_xray_prob = db.Column(db.Float, default=0.0)
    pneumonia_prob = db.Column(db.Float, default=0.0)

# ============================================================================
# FLASK-LOGIN USER LOADER
# ============================================================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_pneumonia_model():
    """Load the trained model"""
    global model
    try:
        model = load_model('checkpoints_3class/best_model_final_3class.keras')
        print("[SUCCESS] Model loaded successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Predict pneumonia from uploaded image"""
    try:
        # Read and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_array = img_resized.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Map to class names
        class_mapping = {0: 'NORMAL', 1: 'NOT_XRAY', 2: 'PNEUMONIA'}
        predicted_label = class_mapping[predicted_class_idx]
        
        return {
            'label': predicted_label,
            'confidence': round(confidence * 100, 2),
            'normal_prob': float(predictions[0][0]),
            'not_xray_prob': float(predictions[0][1]),
            'pneumonia_prob': float(predictions[0][2])
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# ============================================================================
# ROUTES - AUTHENTICATION
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required!', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return render_template('register.html')
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ============================================================================
# ROUTES - MAIN FUNCTIONALITY
# ============================================================================

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.timestamp.desc()).limit(10).all()
    return render_template('dashboard.html', predictions=predictions)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Upload and predict"""
    if request.method == 'POST':
        # Check if files are present
        if 'images' not in request.files:
            flash('No file uploaded!', 'danger')
            return redirect(request.url)
        
        files = request.files.getlist('images')
        predictions = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save file
                filename = secure_filename(file.filename)
                # Add timestamp to avoid duplicates
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Make prediction
                result = predict_image(filepath)
                
                if result:
                    # Save to database
                    prediction = Prediction(
                        filename=filename,
                        prediction=result['label'],
                        confidence=result['confidence'],
                        normal_prob=result['normal_prob'],
                        not_xray_prob=result['not_xray_prob'],
                        pneumonia_prob=result['pneumonia_prob'],
                        user_id=current_user.id
                    )
                    db.session.add(prediction)
                    db.session.commit()
                    
                    predictions.append({
                        'filename': filename,
                        'label': result['label'],
                        'confidence': result['confidence']
                    })
                else:
                    flash(f'Error processing {file.filename}', 'danger')
        
        if predictions:
            flash(f'Successfully processed {len(predictions)} image(s)!', 'success')
            return render_template('predict.html', predictions=predictions)
    
    return render_template('predict.html')

@app.route('/history')
@login_required
def history():
    """View prediction history"""
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/view_prediction/<int:prediction_id>')
@login_required
def view_prediction(prediction_id):
    """View specific prediction details"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Security check - ensure user owns this prediction
    if prediction.user_id != current_user.id:
        flash('Access denied!', 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('view_prediction.html', prediction=prediction)

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(413)
def too_large(e):
    flash('File is too large! Maximum size is 16 MB.', 'danger')
    return redirect(url_for('predict'))

@app.errorhandler(500)
def server_error(e):
    import traceback
    err_msg = traceback.format_exc()
    print("Detailed 500 Error Traceback:")
    print(err_msg)
    return f"<h3>Internal Server Error</h3><pre>{err_msg}</pre>", 500

# ============================================================================
# INITIALIZE DATABASE & LOAD MODEL AT STARTUP (Required for Gunicorn/Render)
# ============================================================================

with app.app_context():
    db.create_all()
    print("[SUCCESS] Database initialized successfully!")

# Load Keras Model
load_pneumonia_model()

# ============================================================================
# RUN (Only used for local testing)
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print(" "*15 + "PNEUMONIA DETECTION WEB APP")
    print("="*70)
    print("\nStarting local Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5000)