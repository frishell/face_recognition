# ====================================================================
# SCRIPT TRAINING SEDERHANA - Face Recognition
# Bisa langsung dijalankan tanpa perlu main.py
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
print("🚀 FACE RECOGNITION TRAINING")
print("="*60)

# ====================================================================
# KONFIGURASI
# ====================================================================

# Sesuaikan dengan dataset Anda
TRAIN_DIR = 'dataset/training'  # Folder training
IMG_SIZE = (100, 100)            # Ukuran gambar
EPOCHS = 30                      # Jumlah epoch (bisa dikurangi untuk testing)
BATCH_SIZE = 32                  # Batch size
VALIDATION_SPLIT = 0.2           # 20% untuk validasi

# Hitung jumlah class (jumlah folder di training)
try:
    num_classes = len([d for d in os.listdir(TRAIN_DIR) 
                      if os.path.isdir(os.path.join(TRAIN_DIR, d))])
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
    # Convolutional Layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    
    # Convolutional Layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Convolutional Layer 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten dan Fully Connected Layer
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model berhasil dibuat!")
print("\n📋 Arsitektur Model:")
model.summary()

# ====================================================================
# DATA AUGMENTATION & GENERATORS
# ====================================================================

print("\n📦 Mempersiapkan data...")

# Data augmentation untuk training
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

# Generator untuk training
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Generator untuk validasi
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\n✅ Data siap:")
print(f"   - Training samples: {train_generator.samples}")
print(f"   - Validation samples: {validation_generator.samples}")
print(f"   - Classes: {num_classes}")

# Simpan class names untuk nanti
class_names = list(train_generator.class_indices.keys())
print(f"\n👥 Daftar orang:")
for i, name in enumerate(class_names[:10], 1):  # Tampilkan 10 pertama
    print(f"   {i}. {name}")
if len(class_names) > 10:
    print(f"   ... dan {len(class_names) - 10} orang lainnya")

# Simpan class_names ke file
with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")
print("\n💾 Class names disimpan ke: class_names.txt")

# ====================================================================
# TRAINING
# ====================================================================

print("\n" + "="*60)
print("🎓 MULAI TRAINING")
print("="*60)
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Ini akan memakan waktu beberapa menit...")
print("="*60 + "\n")

# Hitung steps
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
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

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
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
print(f"\n💾 File yang dibuat:")
print(f"   1. {model_filename} (model trained)")
print(f"   2. training_history.png (grafik)")
print(f"   3. class_names.txt (daftar nama)")
print("\n📝 Langkah selanjutnya:")
print("   - Lihat grafik: training_history.png")
print("   - Testing: python test_model.py")
print("   - Demo realtime: python demo_realtime.py")
print("="*60)