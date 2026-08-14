import cv2
import numpy as np

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

cv2.imshow("Original Image", image)
cv2.imshow("Sobel XY", sobel_xy)

cv2.waitKey(0)
cv2.destroyAllWindows()
