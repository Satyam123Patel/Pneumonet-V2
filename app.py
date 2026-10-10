"""
Complete Pneumonia Detection Web Application
With User Authentication, Prediction History, and Database
"""

import os

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'pneumonet-v2-secure-production-key-9921')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pneumonia_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Dynamic cookie configuration: Secure only when running in production or behind HTTPS
is_dev = os.environ.get('FLASK_ENV') == 'development' or os.environ.get('DEBUG', 'False').lower() in ('true', '1')
app.config['SESSION_COOKIE_SAMESITE'] = 'None' if not is_dev else 'Lax'
app.config['SESSION_COOKIE_SECURE'] = not is_dev

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

def create_pneumonia_model():
    """Create the model architecture matching the trained model"""
    base_model = tf.keras.applications.EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def load_pneumonia_model():
    """Load the trained model with multiple format compatibility"""
    global model
    import sys
    model_path = 'model/best_model_final_3class.keras'

    # Attempt 1: Load complete Keras archive directly (standard Keras 3 / TF 2.16+ method)
    try:
        print("Attempting to load model using tf.keras.models.load_model...", file=sys.stderr, flush=True)
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[SUCCESS] Model loaded successfully via load_model!", file=sys.stderr, flush=True)
        return True
    except Exception as e1:
        print(f"[INFO] load_model could not load directly: {e1}. Falling back to architecture + load_weights...", file=sys.stderr, flush=True)

    # Attempt 2: Recreate architecture and load weights
    try:
        print("Initializing model architecture...", file=sys.stderr, flush=True)
        model = create_pneumonia_model()
        print("Loading weights into model...", file=sys.stderr, flush=True)
        model.load_weights(model_path)
        print("[SUCCESS] Model loaded successfully via load_weights!", file=sys.stderr, flush=True)
        return True
    except Exception as e2:
        import traceback
        print(f"[ERROR] Both model loading methods failed: {e2}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Predict pneumonia from uploaded image"""
    global model
    try:
        # Lazy model loading
        if model is None:
            load_pneumonia_model()
        if model is None:
            print("[ERROR] Model is not loaded")
            return None
            
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
        raise e

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

FREE_GUEST_LIMIT = 3

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Upload and predict with free guest trial support"""
    guest_used = session.get('guest_prediction_count', 0)
    remaining_guest_trials = max(0, FREE_GUEST_LIMIT - guest_used)

    if request.method == 'POST':
        # Check quota for unauthenticated guests
        if not current_user.is_authenticated:
            if remaining_guest_trials <= 0:
                flash(f'You have used all {FREE_GUEST_LIMIT} free guest predictions. Please sign up or login to continue analyzing X-rays.', 'warning')
                return redirect(url_for('register'))

        # Check if files are present
        if 'images' not in request.files:
            flash('No file uploaded!', 'danger')
            return redirect(request.url)
        
        files = request.files.getlist('images')
        predictions = []
        
        for file in files:
            # Check remaining quota per image for guest users
            if not current_user.is_authenticated:
                current_used = session.get('guest_prediction_count', 0)
                if current_used >= FREE_GUEST_LIMIT:
                    flash(f'Free guest trial limit reached ({FREE_GUEST_LIMIT}/{FREE_GUEST_LIMIT}). Please create an account or login for unlimited predictions.', 'warning')
                    break

            if file and allowed_file(file.filename):
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Make prediction
                result = predict_image(filepath)
                
                if result:
                    # Save to database if user is authenticated
                    if current_user.is_authenticated:
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
                    else:
                        # Increment guest prediction count in session
                        session['guest_prediction_count'] = session.get('guest_prediction_count', 0) + 1
                        session.modified = True

                    predictions.append({
                        'filename': filename,
                        'label': result['label'],
                        'confidence': result['confidence']
                    })
                else:
                    flash(f'Error processing {file.filename}', 'danger')
        
        # Recalculate remaining quota after processing
        guest_used = session.get('guest_prediction_count', 0)
        remaining_guest_trials = max(0, FREE_GUEST_LIMIT - guest_used)

        if predictions:
            flash(f'Successfully processed {len(predictions)} image(s)!', 'success')
            return render_template(
                'predict.html', 
                predictions=predictions, 
                free_limit=FREE_GUEST_LIMIT, 
                guest_used=guest_used, 
                remaining_trials=remaining_guest_trials
            )
    
    return render_template(
        'predict.html', 
        free_limit=FREE_GUEST_LIMIT, 
        guest_used=guest_used, 
        remaining_trials=remaining_guest_trials
    )

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
    import sys
    print("Internal Server Error Traceback:", file=sys.stderr)
    print(err_msg, file=sys.stderr)
    return render_template('500.html'), 500

# ============================================================================
# INITIALIZE DATABASE & LOAD MODEL AT STARTUP (Required for Gunicorn/Render)
# ============================================================================

with app.app_context():
    db.create_all()
    print("[SUCCESS] Database initialized successfully!")

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