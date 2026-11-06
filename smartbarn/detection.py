from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class Detection:
    tracker_id: int
    bbox: tuple
    confidence: float

class CowDetectorTracker:
    def __init__(self, weights='yolo11n.pt', tracker='bytetrack.yaml', device='cpu', conf=.12, imgsz=960, cow_class=19):
        self.model=YOLO(weights)
        self.tracker=tracker
        self.device=device
        self.conf=conf
        self.imgsz=imgsz
        self.cow_class=cow_class

    def process(self, frame):
        result=self.model.track(frame,persist=True,tracker=self.tracker,classes=[self.cow_class],conf=self.conf,imgsz=self.imgsz,device=self.device,verbose=False)[0]
        output=[]
        if result.boxes is None or result.boxes.id is None:
            return output
        for box in result.boxes:
            if box.id is None:
                continue
            output.append(Detection(int(box.id.item()),tuple(float(v) for v in box.xyxy[0].cpu().tolist()),float(box.conf.item())))
        return output
