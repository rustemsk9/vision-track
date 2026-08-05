import supervision as sv

class StreamTracker:
    def __init__(self):
        # Tracker for unique IDs across frames
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def update_and_annotate(self, frame, detections):
        # Update tracker with detections
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Annotate frame
        labels = [
            f"#{tracker_id} {confidence:0.2f}"
            for tracker_id, confidence in zip(tracked_detections.tracker_id, tracked_detections.confidence)
        ]
        
        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        
        return annotated_frame, tracked_detections
