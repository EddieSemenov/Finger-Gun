import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break

    # Mirror horizontally
    frame = cv2.flip(frame, 1)

    # Debug labels so we can confirm the image is actually being flipped
    cv2.putText(
        frame,
        "MIRRORED",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
