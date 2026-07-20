import cv2

for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Indice {i}: OK, frame di forma {frame.shape}")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
    cap.release()