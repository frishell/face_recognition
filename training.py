# ====================================================================
# SCRIPT TRAINING FINAL - Face Recognition (CNN)
# Sudah diperbaiki: tanpa error steps_per_epoch = 0
# ====================================================================

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print("="*60)
print("🚀 FACE RECOGNITION TRAINING - FINAL VERSION")
print("="*60)

# ====================================================================
# KONFIGURASI
# ====================================================================

TRAIN_DIR = 'dataset/training'
IMG_SIZE = (100, 100)
EPOCHS = 30
BATCH_SIZE = 8          # Lebih aman untuk dataset kecil
VALIDATION_SPLIT = 0.2  # 20% buat validasi

# Cek jumlah class
try:
    num_classes = len([
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ])
    print(f"\n📊 Jumlah orang terdeteksi: {num_classes}")

    if num_classes == 0:
        print("❌ Error: Tidak ada folder di dataset/training/")
        print("   Jalankan dulu: python extract_faces.py")
        print("   Lalu: python split_dataset.py")
        exit()

except FileNotFoundError:
    print(f"❌ Error: Folder '{TRAIN_DIR}' tidak ditemukan!")
    print("   Buat struktur folder dulu dengan: python setup_project.py")
    exit()

# ====================================================================
# BUILD MODEL CNN
# ====================================================================

print("\n🔨 Membangun model CNN...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model berhasil dibuat!")
model.summary()

# ====================================================================
# DATA AUGMENTATION
# ====================================================================

print("\n📦 Mempersiapkan data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1]
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("\n✅ Data siap:")
print(f"   - Training samples: {train_generator.samples}")
print(f"   - Validation samples: {validation_generator.samples}")
print(f"   - Classes: {num_classes}")

# Simpan nama class
class_names = list(train_generator.class_indices.keys())
with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")
print("💾 Class names disimpan ke: class_names.txt")

# ====================================================================
# TRAINING (AUTO STEPS — tidak akan error)
# ====================================================================

print("\n" + "="*60)
print("🎓 MULAI TRAINING (Final Version)")
print("="*60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

# ====================================================================
# SIMPAN MODEL
# ====================================================================

model_filename = 'face_recognition_model.h5'
model.save(model_filename)
print(f"\n💾 Model disimpan ke: {model_filename}")

# ====================================================================
# VISUALISASI HASIL TRAINING
# ====================================================================

print("\n📊 Membuat grafik hasil training...")

plt.figure(figsize=(12, 4))

# ACC
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# LOSS
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("💾 Grafik disimpan ke: training_history.png")

# ====================================================================
# SUMMARY
# ====================================================================

print("\n" + "="*60)
print("✅ TRAINING SELESAI!")
print("="*60)
print(f"📊 Hasil Akhir:")
print(f"   - Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"   - Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"   - Training Loss: {history.history['loss'][-1]:.4f}")
print(f"   - Validation Loss: {history.history['val_loss'][-1]:.4f}")
print("="*60)
print("\n📝 Langkah selanjutnya:")
print("   - Coba: python test_model.py")
print("   - Untuk realtime webcam: python demo_realtime.py")
print("="*60)
