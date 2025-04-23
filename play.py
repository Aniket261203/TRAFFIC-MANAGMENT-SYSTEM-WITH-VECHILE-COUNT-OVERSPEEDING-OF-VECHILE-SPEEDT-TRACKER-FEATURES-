import cv2                      # OpenCV library for image and video processing
import pandas as pd            # Pandas for data handling (though unused here)
import numpy as np             # NumPy for numerical operations (also unused here)

# Function to get mouse coordinates when mouse moves over the window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # If mouse is moved
        colorsBGR = [x, y]            # Get current coordinates of mouse
        print(colorsBGR)              # Print coordinates to console

# Create a named window to display the video
cv2.namedWindow('TMS')
# Set mouse callback function to capture mouse coordinates on the 'TMS' window
cv2.setMouseCallback('TMS', RGB)

# Load the video file
cap = cv2.VideoCapture('vedio.mp4')  # Make sure the file name is correct

# Infinite loop to read and display video frames
while True:
    ret, frame = cap.read()          # Read a frame from the video
    if not ret:                      # Break loop if no frame is returned (end of video)
        break
    frame = cv2.resize(frame, (1020, 500))  # Resize frame for consistent display
    cv2.imshow("TMS", frame)                # Show the frame in 'TMS' window

    if cv2.waitKey(1) & 0xFF == 27:         # Wait for ESC key to exit
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
