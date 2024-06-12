import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(1)  # 0 for the default camera, you can also pass the camera index if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
