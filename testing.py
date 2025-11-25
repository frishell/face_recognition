# ====================================================================
# SCRIPT TESTING SEDERHANA - Face Recognition
# Bisa langsung dijalankan tanpa perlu main.py
# ====================================================================

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("="*60)
print("🧪 FACE RECOGNITION TESTING")
print("="*60)

# ====================================================================
# KONFIGURASI
# ====================================================================

TEST_DIR = 'dataset/testing'      # Folder testing
MODEL_PATH = 'face_recognition_model.h5'  # Model yang sudah dilatih
IMG_SIZE = (100, 100)
BATCH_SIZE = 32

# ====================================================================
# CEK FILE & FOLDER
# ====================================================================

print("\n🔍 Checking files...")

# Cek model
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model tidak ditemukan!")
    print(f"   File: {MODEL_PATH}")
    print("   Jalankan dulu: python training.py")
    exit()
else:
    print(f"✅ Model ditemukan: {MODEL_PATH}")

# Cek folder testing
if not os.path.exists(TEST_DIR):
    print(f"❌ Error: Folder testing tidak ditemukan!")
    print(f"   Folder: {TEST_DIR}")
    print("   Jalankan dulu: python split_dataset.py")
    exit()
else:
    print(f"✅ Folder testing ditemukan: {TEST_DIR}")

# Hitung jumlah class
try:
    num_classes = len([d for d in os.listdir(TEST_DIR) 
                      if os.path.isdir(os.path.join(TEST_DIR, d))])
    print(f"✅ Jumlah orang: {num_classes}")
    
    if num_classes == 0:
        print("❌ Error: Tidak ada folder di dataset/testing/")
        exit()
        
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ====================================================================
# LOAD MODEL
# ====================================================================

print("\n📦 Loading model...")

try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model berhasil di-load!")
    print(f"\n📋 Model Summary:")
    model.summary()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ====================================================================
# LOAD CLASS NAMES
# ====================================================================

print("\n📝 Loading class names...")

# Coba load dari file
if os.path.exists('class_names.txt'):
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✅ Class names loaded from file")
else:
    # Ambil dari folder testing
    class_names = sorted([d for d in os.listdir(TEST_DIR) 
                         if os.path.isdir(os.path.join(TEST_DIR, d))])
    print(f"✅ Class names dari folder testing")

print(f"\n👥 Daftar orang ({len(class_names)} total):")
for i, name in enumerate(class_names[:10], 1):
    print(f"   {i}. {name}")
if len(class_names) > 10:
    print(f"   ... dan {len(class_names) - 10} orang lainnya")

# ====================================================================
# PREPARE TEST DATA
# ====================================================================

print("\n📦 Mempersiapkan data testing...")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # PENTING: jangan shuffle untuk evaluasi
)

print(f"\n✅ Data testing siap:")
print(f"   - Total samples: {test_generator.samples}")
print(f"   - Classes: {len(test_generator.class_indices)}")
print(f"   - Batch size: {BATCH_SIZE}")

# ====================================================================
# EVALUASI MODEL
# ====================================================================

print("\n" + "="*60)
print("🎯 EVALUASI MODEL")
print("="*60)
print("Sedang memproses... Mohon tunggu...\n")

# Evaluasi
loss, accuracy = model.evaluate(test_generator, verbose=1)

print("\n" + "="*60)
print("📊 HASIL TESTING")
print("="*60)
print(f"Loss:     {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")
print("="*60)

# ====================================================================
# PREDIKSI DETAIL
# ====================================================================

print("\n🔬 Membuat prediksi detail...")

# Reset generator
test_generator.reset()

# Prediksi semua data
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_classes = test_generator.classes

# Hitung statistik
correct = (predicted_classes == true_classes).sum()
total = len(true_classes)
incorrect = total - correct

print(f"\n📈 Statistik Detail:")
print(f"   Total gambar: {total}")
print(f"   Benar: {correct} ({correct/total*100:.2f}%)")
print(f"   Salah: {incorrect} ({incorrect/total*100:.2f}%)")

# ====================================================================
# CONFUSION MATRIX
# ====================================================================

print("\n📊 Membuat confusion matrix...")

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("💾 Confusion matrix disimpan: confusion_matrix.png")

# ====================================================================
# CLASSIFICATION REPORT
# ====================================================================

print("\n📋 Classification Report:")
print("="*60)

report = classification_report(
    true_classes, 
    predicted_classes, 
    target_names=class_names,
    digits=3
)
print(report)

# Simpan ke file
with open('classification_report.txt', 'w') as f:
    f.write("FACE RECOGNITION - CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Total samples: {total}\n")
    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write("="*60 + "\n\n")
    f.write(report)

print("\n💾 Report disimpan: classification_report.txt")

# ====================================================================
# CONTOH PREDIKSI SALAH
# ====================================================================

print("\n🔍 Mencari contoh prediksi yang salah...")

# Cari index yang salah
wrong_indices = np.where(predicted_classes != true_classes)[0]

if len(wrong_indices) > 0:
    print(f"\n❌ Ditemukan {len(wrong_indices)} prediksi salah:")
    
    # Tampilkan max 10 contoh
    num_show = min(10, len(wrong_indices))
    
    for i, idx in enumerate(wrong_indices[:num_show], 1):
        true_label = class_names[true_classes[idx]]
        pred_label = class_names[predicted_classes[idx]]
        confidence = predictions[idx][predicted_classes[idx]] * 100
        
        print(f"\n   {i}. Sample #{idx}")
        print(f"      True:       {true_label}")
        print(f"      Predicted:  {pred_label} ({confidence:.2f}%)")
    
    if len(wrong_indices) > 10:
        print(f"\n   ... dan {len(wrong_indices) - 10} kesalahan lainnya")
    
    # Visualisasi beberapa prediksi salah
    print("\n📸 Membuat visualisasi prediksi salah...")
    
    num_visualize = min(9, len(wrong_indices))
    rows = 3
    cols = 3
    
    plt.figure(figsize=(15, 15))
    
    for i, idx in enumerate(wrong_indices[:num_visualize], 1):
        # Ambil gambar
        img_path = test_generator.filepaths[idx]
        img = plt.imread(img_path)
        
        true_label = class_names[true_classes[idx]]
        pred_label = class_names[predicted_classes[idx]]
        confidence = predictions[idx][predicted_classes[idx]] * 100
        
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                 fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('wrong_predictions.png', dpi=150, bbox_inches='tight')
    print("💾 Visualisasi disimpan: wrong_predictions.png")
    
else:
    print("\n🎉 PERFECT! Semua prediksi benar 100%!")

# ====================================================================
# PER-CLASS ACCURACY
# ====================================================================

print("\n📊 Akurasi Per Orang:")
print("="*60)

for i, class_name in enumerate(class_names):
    class_mask = true_classes == i
    class_total = class_mask.sum()
    
    if class_total > 0:
        class_correct = ((predicted_classes == i) & class_mask).sum()
        class_acc = class_correct / class_total * 100
        
        print(f"{class_name:20s} → {class_correct}/{class_total} ({class_acc:.1f}%)")

# ====================================================================
# CONFIDENCE DISTRIBUTION
# ====================================================================

print("\n📊 Membuat distribusi confidence...")

# Ambil confidence untuk prediksi yang benar
correct_mask = predicted_classes == true_classes
correct_confidences = [predictions[i][predicted_classes[i]] 
                       for i in range(len(predictions)) if correct_mask[i]]

# Ambil confidence untuk prediksi yang salah
wrong_confidences = [predictions[i][predicted_classes[i]] 
                    for i in range(len(predictions)) if not correct_mask[i]]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(correct_confidences, bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title('Confidence - Prediksi Benar', fontsize=12, fontweight='bold')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
if len(wrong_confidences) > 0:
    plt.hist(wrong_confidences, bins=20, color='red', alpha=0.7, edgecolor='black')
    plt.title('Confidence - Prediksi Salah', fontsize=12, fontweight='bold')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Tidak ada\nprediksi salah!', 
             ha='center', va='center', fontsize=16, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
print("💾 Distribusi confidence disimpan: confidence_distribution.png")

# ====================================================================
# SUMMARY
# ====================================================================

print("\n" + "="*60)
print("✅ TESTING SELESAI!")
print("="*60)
print(f"\n📊 Ringkasan:")
print(f"   - Accuracy: {accuracy*100:.2f}%")
print(f"   - Total samples: {total}")
print(f"   - Benar: {correct}")
print(f"   - Salah: {incorrect}")
print("="*60)
print(f"\n💾 File yang dibuat:")
print(f"   1. confusion_matrix.png")
print(f"   2. classification_report.txt")
if len(wrong_indices) > 0:
    print(f"   3. wrong_predictions.png")
print(f"   4. confidence_distribution.png")
print("\n📝 Interpretasi:")
if accuracy >= 0.95:
    print("   🎉 EXCELLENT! Model sangat baik (≥95%)")
elif accuracy >= 0.80:
    print("   ✅ GOOD! Model cukup baik (80-95%)")
    print("   💡 Tips: Tambah data training atau epochs")
else:
    print("   ⚠️  NEEDS IMPROVEMENT! Model perlu perbaikan (<80%)")
    print("   💡 Tips:")
    print("      - Tambah lebih banyak data training")
    print("      - Perbaiki kualitas video/gambar")
    print("      - Pastikan pencahayaan konsisten")
    print("      - Coba training dengan epochs lebih banyak")
print("="*60)