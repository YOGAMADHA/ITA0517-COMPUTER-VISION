import cv2
import numpy as np

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

rows, cols = image.shape[:2]

points1 = np.float32([
    [0, 0],
    [cols - 1, 0],
    [0, rows - 1]
])

points2 = np.float32([
    [50, 50],
    [cols - 100, 30],
    [80, rows - 50]
])

M = cv2.getAffineTransform(points1, points2)

result = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow("Original Image", image)
cv2.imshow("Affine Transformation", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
