import cv2
import numpy as np

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

kernel = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
], dtype=np.float32)

laplacian = cv2.filter2D(image, -1, kernel)

result = cv2.subtract(image, laplacian)

cv2.imshow("Original Image", image)
cv2.imshow("Diagonal Laplacian Sharpening", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
