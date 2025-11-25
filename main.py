import cv2
import numpy as np
from tensorflow.keras.models import load_model

class RealtimeFaceRecognition:
    def __init__(self, model_path, class_names, img_size=(100, 100)):
        self.model = load_model(model_path)
        self.class_names = class_names
        self.img_size = img_size

        # Haarcascade untuk deteksi wajah
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        print("Model dan class names berhasil dimuat!")

    def preprocess_face(self, face):
        """
        Resize wajah lalu normalisasi agar sesuai input model.
        """
        face = cv2.resize(face, self.img_size)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        return face

    def recognize_from_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Tidak dapat membuka webcam.")
            return

        print("🎥 Webcam aktif. Tekan 'q' untuk keluar.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]

                # Preprocess
                processed = self.preprocess_face(face_img)

                # Prediksi
                prediction = self.model.predict(processed, verbose=0)
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id] * 100

                name = self.class_names[class_id]

                # Gambar bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Tulis hasil
                cv2.putText(frame,
                            f"{name} ({confidence:.1f}%)",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2)

            cv2.imshow("Realtime Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
