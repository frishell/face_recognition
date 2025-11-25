import os
import shutil
import random

def split_dataset(train_ratio=0.8):
    """
    Split dataset menjadi training dan testing
    
    Args:
        train_ratio: Proporsi data training (0.8 = 80% training, 20% testing)
    """
    
    print("="*60)
    print("📊 SPLIT DATASET: TRAINING & TESTING")
    print("="*60)
    
    temp_folder = 'dataset/temp'
    
    # Cek apakah folder temp ada
    if not os.path.exists(temp_folder):
        print("❌ Folder dataset/temp tidak ditemukan!")
        print("   Jalankan dulu: python extract_faces.py")
        return
    
    # Ambil semua folder person
    persons = [d for d in os.listdir(temp_folder) 
               if os.path.isdir(os.path.join(temp_folder, d))]
    
    if not persons:
        print("❌ Tidak ada data di folder dataset/temp")
        return
    
    print(f"\n📂 Ditemukan {len(persons)} orang:\n")
    
    total_train = 0
    total_test = 0
    
    for person in persons:
        person_temp = os.path.join(temp_folder, person)
        person_train = f'dataset/training/{person}'
        person_test = f'dataset/testing/{person}'
        
        # Buat folder
        os.makedirs(person_train, exist_ok=True)
        os.makedirs(person_test, exist_ok=True)
        
        # Ambil semua gambar
        images = [f for f in os.listdir(person_temp) if f.endswith('.jpg')]
        random.shuffle(images)
        
        # Hitung split
        num_train = int(len(images) * train_ratio)
        train_images = images[:num_train]
        test_images = images[num_train:]
        
        # Copy ke training
        for img in train_images:
            src = os.path.join(person_temp, img)
            dst = os.path.join(person_train, img)
            shutil.copy(src, dst)
        
        # Copy ke testing
        for img in test_images:
            src = os.path.join(person_temp, img)
            dst = os.path.join(person_test, img)
            shutil.copy(src, dst)
        
        total_train += len(train_images)
        total_test += len(test_images)
        
        print(f"   ✓ {person:20s} → Train: {len(train_images):2d}, Test: {len(test_images):2d}")
    
    print("\n" + "="*60)
    print(f"✅ SELESAI!")
    print(f"   Total Training: {total_train} gambar")
    print(f"   Total Testing:  {total_test} gambar")
    print(f"   Ratio: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%")
    print("="*60)
    print("\n📁 Dataset siap di:")
    print("   - dataset/training/")
    print("   - dataset/testing/")
    print("\n📝 Langkah selanjutnya:")
    print("   Jalankan training: python train_model.py")


if __name__ == "__main__":
    split_dataset(train_ratio=0.8)  # 80% training, 20% testing