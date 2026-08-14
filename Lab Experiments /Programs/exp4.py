import cv2
import numpy as np

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.uint8)

dilated = cv2.dilate(gray, kernel, iterations=1)

cv2.imshow("Original Image", image)
cv2.imshow("Dilated Image", dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()
