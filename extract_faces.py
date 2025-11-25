import cv2
import os
from pathlib import Path

class SimpleFaceExtractor:
    def __init__(self):
        # Load Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def extract_from_video(self, video_path, person_name, 
                          num_images=30, output_size=(100, 100)):
        """
        Ekstrak wajah dari video
        
        Args:
            video_path: Path ke file video
            person_name: Nama orang (untuk nama folder)
            num_images: Jumlah gambar yang ingin diekstrak
            output_size: Ukuran output gambar (width, height)
        """
        
        # Buat folder output
        output_folder = f'dataset/temp/{person_name}'
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Buka video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Tidak bisa membuka video {video_path}")
            return 0
        
        # Hitung total frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // num_images)
        
        print(f"\n📹 Memproses: {person_name}")
        print(f"   Video: {video_path}")
        print(f"   Total frames: {total_frames}")
        print(f"   Akan ekstrak: {num_images} gambar")
        
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened() and saved_count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Proses setiap N frame
            if frame_count % frame_interval == 0:
                # Convert ke grayscale untuk deteksi
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Deteksi wajah
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50)
                )
                
                # Jika ada wajah terdeteksi
                if len(faces) > 0:
                    # Ambil wajah terbesar (asumsi: wajah utama)
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    
                    # Crop wajah dengan margin
                    margin = 20
                    y1 = max(0, y - margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    x1 = max(0, x - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Resize
                    face_img = cv2.resize(face_img, output_size)
                    
                    # Simpan
                    filename = f"{output_folder}/face_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, face_img)
                    saved_count += 1
                    
                    # Progress
                    if saved_count % 5 == 0:
                        print(f"   Progress: {saved_count}/{num_images}")
            
            frame_count += 1
        
        cap.release()
        print(f"✅ Selesai: {saved_count} gambar tersimpan\n")
        return saved_count


def main():
    print("="*60)
    print("🎯 EKSTRAKSI WAJAH DARI VIDEO")
    print("="*60)
    
    extractor = SimpleFaceExtractor()
    
    # Daftar video di folder videos/
    video_folder = 'videos'
    video_files = [f for f in os.listdir(video_folder) 
                   if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print("❌ Tidak ada video di folder 'videos/'")
        return
    
    print(f"\n📂 Ditemukan {len(video_files)} video:\n")
    for i, video in enumerate(video_files, 1):
        print(f"   {i}. {video}")
    
    print("\n" + "="*60)
    print("Mulai ekstraksi...\n")
    
    # Proses setiap video
    total_extracted = 0
    for video_file in video_files:
        # Ambil nama dari filename (tanpa extension)
        person_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_folder, video_file)
        
        # Ekstrak
        count = extractor.extract_from_video(
            video_path=video_path,
            person_name=person_name,
            num_images=30  # Total 30 gambar per orang
        )
        total_extracted += count
    
    print("="*60)
    print(f"✅ SELESAI!")
    print(f"   Total gambar diekstrak: {total_extracted}")
    print(f"   Lokasi: dataset/temp/")
    print("="*60)
    print("\n📝 Langkah selanjutnya:")
    print("   Jalankan: python split_dataset.py")


if __name__ == "__main__":
    main()