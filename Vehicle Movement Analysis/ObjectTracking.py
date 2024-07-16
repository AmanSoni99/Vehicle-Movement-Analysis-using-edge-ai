from sort import Sort  # Make sure to have SORT algorithm implementation available

tracker = Sort()

# Update object tracking for each frame
def update_tracking(frame, detected_objects):
    tracked_objects = tracker.update(detected_objects)
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        cv2.rectangle(frame, (x1, y1), 
        (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Integration with detection code
while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, objects = detect_objects(frame)
    tracked_frame = update_tracking(processed_frame, objects)

    cv2.imshow('Object Detection and Tracking', tracked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
