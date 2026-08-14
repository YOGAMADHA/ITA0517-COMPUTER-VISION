import cv2

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

result = image.copy()

cv2.rectangle(
    result,
    (50, 50),
    (300, 300),
    (0, 255, 0),
    2
)

object_image = image[50:300, 50:300]

cv2.imshow("Image with Rectangle", result)
cv2.imshow("Extracted Object", object_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
