"""Motion/activity proxies; posture only when user supplies a trained classification model."""
from collections import defaultdict,deque
import numpy as np

class ActivityAnalyzer:
    def __init__(self, fps, window_seconds=2.0, movement_threshold_px_s=12.0, posture_model_path=None):
        self.history=defaultdict(lambda:deque(maxlen=max(4,int(fps*window_seconds))))
        self.threshold=movement_threshold_px_s
        self.classifier=None
        if posture_model_path:
            from ultralytics import YOLO
            self.classifier=YOLO(posture_model_path)

    def update(self,cow_id,center,time_s,frame=None,bbox=None):
        history=self.history[cow_id]
        history.append((time_s,np.array(center,dtype=float)))
        speed=0.0
        if len(history)>=2:
            dt=history[-1][0]-history[0][0]
            if dt>0: speed=float(np.linalg.norm(history[-1][1]-history[0][1])/dt)
        activity='moving' if speed>=self.threshold else 'low_motion'
        posture='unknown'
        if self.classifier is not None and frame is not None and bbox is not None:
            h,w=frame.shape[:2];x1,y1,x2,y2=map(int,bbox)
            crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
            if crop.size:
                result=self.classifier.predict(crop,verbose=False)[0]
                if result.probs is not None:
                    posture=result.names[int(result.probs.top1)]
        return dict(activity=activity,motion_px_s=round(speed,2),posture=posture)
