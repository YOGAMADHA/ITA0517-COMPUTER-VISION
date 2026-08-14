import cv2
import numpy as np

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

result = cv2.filter2D(image, -1, kernel)

cv2.imshow("Original Image", image)
cv2.imshow("Positive Center Sharpening", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
