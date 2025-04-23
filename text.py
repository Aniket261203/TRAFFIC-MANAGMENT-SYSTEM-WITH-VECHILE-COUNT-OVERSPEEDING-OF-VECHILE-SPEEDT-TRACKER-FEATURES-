import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
from tracker import *  # Assuming 'Tracker' is a custom object tracking class
from twilio.rest import Client  # Import Twilio's Client for sending SMS

# Load YOLO model for object detection
model = YOLO('yolov8s.pt')

# Mouse coordinate tool (optional, used for debugging or user input)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])  # Print the mouse coordinates on the frame

cv2.namedWindow('TMS')  # Create a window named 'TMS'
cv2.setMouseCallback('TMS', RGB)  # Set mouse callback for the window

# Open video file for processing
cap = cv2.VideoCapture('vedio.mp4')

# Load class labels (e.g., car, truck, etc.) from the coco dataset
with open("coco.txt", "r") as f:
    class_list = f.read().splitlines()

# Define Region of Interests (ROIs) for tracking vehicles in two areas
area = [(170, 219), (1018, 219), (1003, 286), (110, 286)]
area2 = [(177, 320), (1000, 320), (1011, 370), (169, 370)]

# Initialize tracker and data storage for vehicle info
tracker = Tracker()
vehicles_entering = {}  # Stores entry time of vehicles in area 1
vehicles_speed = {}  # Stores the calculated speed of vehicles
area_c = set()  # Stores unique vehicle IDs
violators = []  # Stores overspeeding vehicle information

# Parameters
speed_limit = 10  # Speed limit in km/h
distance_m = 25  # Distance between area1 and area2 in meters

count = 0  # Frame counter for processing

# Function to send SMS notification
def send_sms(violator_data):
    # Twilio settings
    account_sid = "AC8f104987fe81837a7d3d3d2663f958f2"  # Replace with your Twilio Account SID
    auth_token = "3e6a380ab15d5d6365e0880230f4486e"  # Replace with your Twilio Auth Token
    from_phone_number = "+15856343197"  # Replace with your Twilio phone number (e.g., +1xxxxxxxxxx)
    to_phone_number = "+916388644038"  # Replace with the recipient's phone number (e.g., +1xxxxxxxxxx)

    # Create Twilio client
    client = Client(account_sid, auth_token)

    # SMS content
    message_body = f"""
    Overspeeding Alert:
    Vehicle ID: {violator_data['Vehicle ID']}
    Vehicle Type: {violator_data['Vehicle Type']}
    Speed: {violator_data['Speed (km/h)']} km/h
    Timestamp: {violator_data['Time']}
    Violation ID: {violator_data['Violation ID']}
    """
    
    try:
        # Send the SMS
        message = client.messages.create(
            body=message_body,
            from_=from_phone_number,
            to=to_phone_number
        )
        print("SMS sent successfully.")
    except Exception as e:
        print(f"Error sending SMS: {e}")

# Main loop for video processing
while True:
    ret, frame = cap.read()
    if not ret:
        break  # If the frame could not be read, stop the loop

    count += 1
    if count % 3 != 0:  # Process every 3rd frame for efficiency
        continue

    frame = cv2.resize(frame, (1020, 500))  # Resize the frame for consistency

    # Run YOLO model to detect objects
    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes  # Get detected bounding boxes from YOLO

    # Convert bounding box data to numpy arrays
    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else []

    detections = []  # Store valid detections (vehicle bounding boxes)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])  # Extract coordinates of the bounding box
        class_id = int(cls[i])  # Get the class ID of the detected object
        class_name = class_list[class_id]  # Map class ID to class name
        if class_name in ['car', 'truck', 'bus', 'motorcycle']:  # Only track vehicles
            detections.append([x1, y1, x2, y2])  # Store the bounding box of the vehicle

    tracked_objects = tracker.update(detections)  # Update the tracker with new detections

    for bbox in tracked_objects:
        x3, y3, x4, y4, obj_id = bbox  # Extract bounding box coordinates and object ID
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2  # Calculate the center of the bounding box

        # Check if the center of the vehicle is within area 1 and area 2
        in_area1 = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0
        in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0

        # Draw the bounding box around the vehicle
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
        cv2.putText(frame, f'ID:{obj_id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Mark entry time for the vehicle when it enters area 1
        if in_area1 and obj_id not in vehicles_entering:
            vehicles_entering[obj_id] = time.time()  # Record the time the vehicle entered area 1

        # Speed calculation when the vehicle enters area 2
        if in_area2:
            if obj_id in vehicles_entering and obj_id not in vehicles_speed:
                elapsed_time = time.time() - vehicles_entering[obj_id]  # Calculate elapsed time
                try:
                    # Calculate speed in km/h (distance/time)
                    speed = (distance_m / elapsed_time) * 3.6  # Convert m/s to km/h
                    vehicles_speed[obj_id] = int(speed)  # Store the speed of the vehicle
                    area_c.add(obj_id)  # Add the vehicle ID to the set of vehicles in area 2

                    # Display the calculated speed on the frame
                    cv2.putText(frame, f'{int(speed)} km/h', (x3, y4 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Check if the vehicle is overspeeding
                    if speed > speed_limit:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Get current timestamp
                        
                        # Add the violator details to the list
                        violators.append({
                            "Vehicle ID": obj_id,  # Unique vehicle ID
                            "Vehicle Type": class_name,  # Type of the vehicle (car, truck, etc.)
                            "Speed (km/h)": int(speed),  # Speed of the vehicle
                            "Time": timestamp,  # Timestamp of the violation
                            "Violation ID": f"VIO-{obj_id}-{timestamp}",  # Unique violation ID
                        })
                        
                        # Send SMS notification about the overspeeding violation
                        send_sms(violators[-1])  # Send the last added violator details

                        # Show "OVERSPEED!" text on the frame
                        cv2.putText(frame, "OVERSPEED!", (cx, cy - 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

                except ZeroDivisionError:
                    pass  # Skip if elapsed time is zero due to very small movement
            else:
                # Display speed if vehicle has already been tracked in area 2
                if obj_id in vehicles_speed:
                    speed = vehicles_speed[obj_id]
                    cv2.putText(frame, f'{speed} km/h', (x3, y4 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw areas on the frame (area 1 and area 2)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)

    # Display the vehicle count in area 2
    cv2.putText(frame, f'Vehicle Count: {len(area_c)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame in the window
    cv2.imshow("TMS", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit when the 'Esc' key is pressed
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows

# Save overspeeding data to a CSV file
if violators:
    # Convert violators data into a DataFrame and save to CSV
    df = pd.DataFrame(violators)
    df.to_csv("overspeed_violations.csv", index=False)  # Save the DataFrame to a CSV file
    print("Overspeed violations saved to overspeed_violations.csv")
else:
    print("No overspeeding vehicles detected.")
