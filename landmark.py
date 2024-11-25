import cv2
import dlib
import matplotlib.pyplot as plt

# โหลดโมเดลตรวจจับใบหน้าและจุด landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/68_face_landmarks.dat") #shape_predictor_68_face_landmarks.dat

# อ่านภาพ
image_path = "Image.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # แปลงภาพเป็น RGB เพื่อการแสดงผล
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # แปลงภาพเป็นขาวดำ (Grayscale)

# ขั้นตอนที่ 1: แสดงภาพต้นฉบับและภาพขาวดำ
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# ตรวจจับใบหน้าในภาพ
faces = detector(gray)

# ขั้นตอนที่ 2: วาดกรอบรอบใบหน้าที่ตรวจจับได้
img_faces = img_rgb.copy()
for face in faces:
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cv2.rectangle(img_faces, (x, y), (x + w, y + h), (255, 0, 0), 2)  # วาดกรอบสีน้ำเงินรอบใบหน้า

plt.figure(figsize=(8, 6))
plt.imshow(img_faces)
plt.title("Detected Faces with Bounding Boxes")
plt.axis("off")
plt.show()

# ขั้นตอนที่ 3: ตรวจจับจุด landmark และวาดบนภาพ
for face in faces:
    # ตรวจจับจุด landmark บนใบหน้า
    landmarks = predictor(gray, face)
    
    # วาดจุด landmark (68 จุด)
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img_rgb, (x, y), 2, (0, 255, 0), -1)  # วาดจุดสีเขียวบนจุด landmark

# แสดงภาพพร้อมจุด landmark
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title("Face Landmarks (68 Points)")
plt.axis("off")
plt.show()
