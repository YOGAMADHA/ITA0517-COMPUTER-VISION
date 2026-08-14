import cv2

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

rows, cols = image.shape[:2]

crop = image[
    rows // 4:rows // 2,
    cols // 4:cols // 2
]

result = image.copy()

h, w = crop.shape[:2]

result[10:10+h, 10:10+w] = crop

cv2.imshow("Original Image", image)
cv2.imshow("Cropped and Pasted Image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
