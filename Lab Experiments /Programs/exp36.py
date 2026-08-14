import cv2

image = cv2.imread(r"C:\Users\yogar\Downloads\Computer Vision\Picture1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (5, 5), 0)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    1.2,
    50,
    param1=100,
    param2=50,
    minRadius=10,
    maxRadius=300
)

result = image.copy()

if circles is not None:
    circles = circles[0]

    for circle in circles:
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])

        cv2.circle(result, (x, y), r, (0, 255, 0), 2)
        cv2.putText(
            result,
            "Watch",
            (x - 30, y - r),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

cv2.imshow("Watch Recognition", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
