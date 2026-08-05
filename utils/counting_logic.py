import supervision as sv
import numpy as np
import cv2

class ROICounter:
    def __init__(self, polygon):
        # Polygon defines the Region of Interest
        self.polygon_pts = np.array(polygon, dtype=np.int32)
        self.zone = sv.PolygonZone(polygon=self.polygon_pts)

    def process_and_annotate(self, frame, detections):
        # Trigger zone logic
        mask = self.zone.trigger(detections=detections)
        
        # Annotate zone on frame manually to avoid supervision/numpy2.0 crash
        annotated_frame = frame.copy()
        cv2.polylines(annotated_frame, [self.polygon_pts], isClosed=True, color=(255, 255, 255), thickness=2)
        
        return annotated_frame, sum(mask)
