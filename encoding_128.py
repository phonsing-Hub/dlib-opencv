import cv2
import dlib
import numpy as np

# โหลดโมเดลตรวจจับใบหน้าและจุด landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/68_face_landmarks.dat") #shape_predictor_68_face_landmarks.dat
face_rec_model = dlib.face_recognition_model_v1("models/face_model_v1.dat") #dlib_face_recognition_resnet_model_v1.dat

# อ่านภาพ
img = cv2.imread("Image.jpg")
# ทำให้ภาพเบลอโดยใช้ Gaussian blur
# blurred_image = cv2.GaussianBlur(img, (15, 15), 6)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ตรวจจับใบหน้าในภาพ
faces = detector(gray)

for face in faces:
    # ตรวจจับจุด landmark บนใบหน้า
    landmarks = predictor(gray, face)
    
    # วาดจุด landmark บนใบหน้า (68 จุด)
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    
    # สร้าง Face Encoding โดยใช้เวกเตอร์ 128 ค่า
    encoding = np.array(face_rec_model.compute_face_descriptor(img, landmarks))

    print("Face Encoding (128-d vector)")
    print(encoding)
    print("\n")
# แสดงภาพที่มีจุด landmark
cv2.imshow("Landmarks", img)
while True:
    cv2.imshow("Landmarks", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
            break
    
cv2.destroyAllWindows()
