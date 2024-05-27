import cv2

# Replace 1 with the actual camera index you want to check
cap = cv2.VideoCapture(0)
num_cameras = cap.isOpened()

if num_cameras:
  print(f"Camera {1} is available")
else:
  print(f"Camera {1} is not available")

# Release the camera object
cap.release()
