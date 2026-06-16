import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
import cv2
import shutil
import random
import json

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

IMG_SIZE = 224
BATCH_SIZE = 32 # Larger batch size for faster epoch processing on CPU
LEARNING_RATE = 1e-4

ORIGINAL_DATA_DIR = r"C:\Users\Lenovo\Downloads\chest_xray"
NEW_DATA_DIR = r"C:\Desktop\pneumonia detectino\pneumonia detection\chest_xray_3class"
CHECKPOINT_DIR = r"C:\Desktop\pneumonia detectino\pneumonia detection\checkpoints_3class"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def generate_synthetic_images(output_dir, num_images):
    """Create synthetic images that are clearly NOT X-rays"""
    os.makedirs(output_dir, exist_ok=True)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i in range(num_images):
        img_type = i % 5
        
        if img_type == 0:  # Colorful gradient
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            for x in range(IMG_SIZE):
                for y in range(IMG_SIZE):
                    img[x, y] = [x % 256, y % 256, (x+y) % 256]
        
        elif img_type == 1:  # Random noise with structures
            img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (15, 15), 0)
        
        elif img_type == 2:  # Geometric patterns
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            for j in range(12):
                pt1 = (np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE))
                pt2 = (np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE))
                color = colors[np.random.randint(0, len(colors))]
                cv2.line(img, pt1, pt2, color, np.random.randint(2, 8))
        
        elif img_type == 3:  # Circles and shapes
            img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
            for j in range(6):
                center = (np.random.randint(30, IMG_SIZE - 30), np.random.randint(30, IMG_SIZE - 30))
                radius = np.random.randint(10, 45)
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.circle(img, center, radius, color, -1)
        
        else:  # Text-like patterns
            img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 200
            for j in range(25):
                pt1 = (np.random.randint(10, IMG_SIZE - 20), np.random.randint(10, IMG_SIZE - 20))
                text = chr(65 + np.random.randint(0, 26))
                cv2.putText(img, text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        img_path = os.path.join(output_dir, f"not_xray_{i:04d}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def create_dataset():
    print("Preparing 3-class dataset...")
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA', 'NOT_XRAY']
    
    for s in splits:
        for c in classes:
            os.makedirs(os.path.join(NEW_DATA_DIR, s, c), exist_ok=True)
            
    # Copy from original dataset test set directly to NEW_DATA_DIR test set
    for c in ['NORMAL', 'PNEUMONIA']:
        src = os.path.join(ORIGINAL_DATA_DIR, 'test', c)
        dst = os.path.join(NEW_DATA_DIR, 'test', c)
        if os.path.exists(src):
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                
    # Copy train images and split 10% into val set
    for c in ['NORMAL', 'PNEUMONIA']:
        src = os.path.join(ORIGINAL_DATA_DIR, 'train', c)
        if os.path.exists(src):
            all_files = os.listdir(src)
            random.shuffle(all_files)
            val_size = int(len(all_files) * 0.10)
            val_files = all_files[:val_size]
            train_files = all_files[val_size:]
            
            # Copy to new train directory
            for f in train_files:
                shutil.copy2(os.path.join(src, f), os.path.join(NEW_DATA_DIR, 'train', c, f))
            # Copy to new val directory
            for f in val_files:
                shutil.copy2(os.path.join(src, f), os.path.join(NEW_DATA_DIR, 'val', c, f))

    # Generate synthetic NOT_XRAY images
    print("Generating NOT_XRAY images...")
    generate_synthetic_images(os.path.join(NEW_DATA_DIR, 'train', 'NOT_XRAY'), 800)
    generate_synthetic_images(os.path.join(NEW_DATA_DIR, 'val', 'NOT_XRAY'), 100)
    generate_synthetic_images(os.path.join(NEW_DATA_DIR, 'test', 'NOT_XRAY'), 150)
    
    print("Dataset setup completed successfully!")

if not os.path.exists(NEW_DATA_DIR) or len(os.listdir(NEW_DATA_DIR)) == 0:
    create_dataset()
else:
    print("Dataset already reorganized.")

# ============================================================================
# TRAIN MODEL
# ============================================================================

# Data generators - NOTE: NO rescale=1./255 because EfficientNetB0 has internal rescaling!
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    os.path.join(NEW_DATA_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(NEW_DATA_DIR, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(NEW_DATA_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Class mapping: {train_generator.class_indices}")

# Build Model
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
outputs = Dense(3, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Compute class weights
class_indices = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_indices),
    y=class_indices
)
class_weight_dict = dict(enumerate(class_weights))

print(f"Class weights: {class_weight_dict}")

# Callbacks
model_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model_final_3class.keras')
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1),
    ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("Starting training (Phase 1: Head training)...")
history = model.fit(
    train_generator,
    epochs=8, # 8 epochs is fast on CPU and highly accurate with proper scaling
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Save final model also as general name
model.save(os.path.join(CHECKPOINT_DIR, 'best_model_final_3class.keras'))
print("Model trained and saved successfully!")
