import cv2

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

cv2.putText(
    image,
    "COMPUTER VISION",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.imshow("Watermarked Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
