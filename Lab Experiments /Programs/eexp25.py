import cv2

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

gradient = cv2.magnitude(gx, gy)

gradient = cv2.convertScaleAbs(gradient)

result = cv2.add(gray, gradient)

cv2.imshow("Original Image", image)
cv2.imshow("Gradient Masking", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
