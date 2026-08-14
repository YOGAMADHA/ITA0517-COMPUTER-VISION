import cv2

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel_y = cv2.convertScaleAbs(sobel_y)

cv2.imshow("Original Image", image)
cv2.imshow("Sobel Y", sobel_y)

cv2.waitKey(0)
cv2.destroyAllWindows()
