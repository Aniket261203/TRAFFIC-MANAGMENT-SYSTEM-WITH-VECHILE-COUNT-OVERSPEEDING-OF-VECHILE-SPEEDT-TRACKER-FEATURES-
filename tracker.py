class Tracker:
    def __init__(self):
        # Initialize a dictionary to store the center points of tracked objects
        self.center_points = {}
        # Initialize a counter for assigning unique IDs to new objects
        self.id_count = 0

    def update(self, objects_rect):
        # List to store bounding boxes along with their assigned IDs
        objects_bbs_ids = []
        
        # Iterate over each detected object's bounding box
        for rect in objects_rect:
            x, y, w, h = rect  # Unpack the bounding box coordinates
            # Calculate the center of the bounding box
            cx = (x + w) // 2
            cy = (y + h) // 2

            same_object_detected = False  # Flag to check if the object is already being tracked
            
            # Iterate over existing tracked objects to find a match
            for id, pt in self.center_points.items():
                # Calculate the Euclidean distance between the current object and the tracked object
                dist = ((cx - pt[0]) ** 2 + (cy - pt[1]) ** 2) ** 0.5
                # If the distance is below a threshold, consider it the same object
                if dist < 50:
                    # Update the center point of the tracked object
                    self.center_points[id] = (cx, cy)
                    # Append the bounding box and ID to the list
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True  # Set the flag to True
                    break  # Exit the loop as the object is already tracked

            # If the object is not already tracked, assign a new ID
            if not same_object_detected:
                # Add the new object to the center points dictionary
                self.center_points[self.id_count] = (cx, cy)
                # Append the bounding box and new ID to the list
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                # Increment the ID counter for the next new object
                self.id_count += 1

        # Return the list of bounding boxes with their assigned IDs
        return objects_bbs_ids
