from main import RealtimeFaceRecognition

# Nama orang sesuai urutan folder (alfabetis)
class_names = ['andi', 'budi', 'citra', 'dedi', 'eka']

recognizer = RealtimeFaceRecognition(
    model_path='my_face_model.h5',
    class_names=class_names
)

# Test dengan webcam
recognizer.recognize_from_webcam()