import cv2
import numpy as np

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

rows, cols = image.shape[:2]

points1 = np.float32([
    [0, 0],
    [cols - 1, 0],
    [cols - 1, rows - 1],
    [0, rows - 1]
])

points2 = np.float32([
    [50, 40],
    [cols - 80, 20],
    [cols - 40, rows - 40],
    [30, rows - 20]
])

H, status = cv2.findHomography(points1, points2)

result = cv2.warpPerspective(image, H, (cols, rows))

cv2.imshow("Original Image", image)
cv2.imshow("Homography", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
