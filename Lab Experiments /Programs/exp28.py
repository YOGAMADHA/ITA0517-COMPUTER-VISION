import cv2
import numpy as np

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

boundary = cv2.filter2D(gray, -1, kernel)

cv2.imshow("Original Image", image)
cv2.imshow("Boundary", boundary)

cv2.waitKey(0)
cv2.destroyAllWindows()
