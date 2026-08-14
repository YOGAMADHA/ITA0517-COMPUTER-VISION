import cv2

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

blur = cv2.GaussianBlur(image, (7, 7), 0)

mask = cv2.subtract(image, blur)

result = cv2.add(image, mask)

cv2.imshow("Original Image", image)
cv2.imshow("Unsharp Masking", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
