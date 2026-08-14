import cv2

video_path = input("Enter video path: ")

cap = cv2.VideoCapture(video_path)

frames = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frames.append(frame)

cap.release()

for frame in reversed(frames):

    cv2.imshow("Reverse Video", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
