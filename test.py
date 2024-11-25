import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog

# โหลดภาพ
image_path = "images/me.png"  # ใส่ path ของภาพที่ต้องการ
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# แสดงภาพต้นฉบับและภาพขาวดำ
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# การแบ่งภาพเป็นเซลล์และการคำนวณ HOG
cell_size = (8, 8)  # ขนาดของเซลล์
block_size = (2, 2)  # ขนาดของบล็อก (กี่เซลล์ต่อบล็อก)
nbins = 9  # จำนวน bins สำหรับทิศทางกราดิเอนต์

# คำนวณ HOG features
hog_features, hog_image = hog(
    image_gray,
    orientations=nbins,
    pixels_per_cell=cell_size,
    cells_per_block=block_size,
    block_norm="L2-Hys",
    visualize=True,
    feature_vector=True,
)

# แสดงภาพ HOG
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_gray)
plt.title("Image (for comparison)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_image)
plt.title("HOG Image")
plt.axis("off")
plt.show()

# แสดงเวกเตอร์คุณลักษณะ (Feature Vector)
plt.figure(figsize=(12, 4))
plt.plot(hog_features, color='blue')
plt.title("Feature Vector from HOG")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.show()

