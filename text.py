import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
from tracker import *

# Load YOLO model
model = YOLO('yolov8s.pt')

# Mouse coordinate tool (optional)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow('TMS')
cv2.setMouseCallback('TMS', RGB)

# Open video file
cap = cv2.VideoCapture('vedio.mp4')

# Load class labels
with open("coco.txt", "r") as f:
    class_list = f.read().splitlines()

# Define ROIs
area = [(170, 219), (1018, 219), (1003, 286), (110, 286)]
area2 = [(177, 320), (1000, 320), (1011, 370), (169, 370)]

# Tracker and vehicle data stores
tracker = Tracker()
vehicles_entering = {}
vehicles_speed = {}
area_c = set()
violators = []  # Track overspeeders

# Parameters
speed_limit = 40  # km/h
distance_m = 25

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes

    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else []

    detections = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        class_id = int(cls[i])
        class_name = class_list[class_id]
        if class_name in ['car', 'truck', 'bus', 'motorcycle']:
            detections.append([x1, y1, x2, y2])

    tracked_objects = tracker.update(detections)

    for bbox in tracked_objects:
        x3, y3, x4, y4, obj_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        in_area1 = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0
        in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0

        # Draw bounding box
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
        cv2.putText(frame, f'ID:{obj_id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Mark entry time
        if in_area1 and obj_id not in vehicles_entering:
            vehicles_entering[obj_id] = time.time()

        # Speed calculation
        if in_area2:
            if obj_id in vehicles_entering and obj_id not in vehicles_speed:
                elapsed_time = time.time() - vehicles_entering[obj_id]
                try:
                    speed = (distance_m / elapsed_time) * 3.6  # m/s to km/h
                    vehicles_speed[obj_id] = int(speed)
                    area_c.add(obj_id)

                    # Display speed
                    cv2.putText(frame, f'{int(speed)} km/h', (x3, y4 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Overspeed check
                    if speed > speed_limit:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        violators.append({
                            "Vehicle ID": obj_id,
                            "Speed (km/h)": int(speed),
                            "Time": timestamp
                        })
                        cv2.putText(frame, "OVERSPEED!", (cx, cy - 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

                except ZeroDivisionError:
                    pass
            else:
                if obj_id in vehicles_speed:
                    speed = vehicles_speed[obj_id]
                    cv2.putText(frame, f'{speed} km/h', (x3, y4 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw areas
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)

    # Vehicle count
    cv2.putText(frame, f'Vehicle Count: {len(area_c)}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("TMS", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Export overspeeding data
if violators:
    df = pd.DataFrame(violators)
    df.to_csv("overspeed_violations.csv", index=False)
    print("Overspeed violations saved to overspeed_violations.csv")
else:
    print("No overspeeding vehicles detected.")
