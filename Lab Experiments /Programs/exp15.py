import cv2
import numpy as np

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

rows, cols = image.shape[:2]

source = np.float32([
    [0, 0],
    [cols - 1, 0],
    [cols - 1, rows - 1],
    [0, rows - 1]
])

destination = np.float32([
    [30, 30],
    [cols - 50, 20],
    [cols - 20, rows - 30],
    [40, rows - 20]
])

H, status = cv2.findHomography(source, destination)

result = cv2.warpPerspective(image, H, (cols, rows))

cv2.imshow("Original Image", image)
cv2.imshow("Direct Linear Transformation", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
