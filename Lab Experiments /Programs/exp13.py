import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rows, cols = frame.shape[:2]

    points1 = np.float32([
        [0, 0],
        [cols - 1, 0],
        [cols - 1, rows - 1],
        [0, rows - 1]
    ])

    points2 = np.float32([
        [50, 50],
        [cols - 100, 30],
        [cols - 50, rows - 50],
        [50, rows - 30]
    ])

    M = cv2.getPerspectiveTransform(points1, points2)

    result = cv2.warpPerspective(frame, M, (cols, rows))

    cv2.imshow("Perspective Video", result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
