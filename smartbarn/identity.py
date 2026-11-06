"""Experimental short-gap ID reconciliation, not a trained animal re-ID system."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class IdentityState:
    stable_id: int
    raw_id: int
    center: np.ndarray
    velocity: np.ndarray
    hist: np.ndarray | None
    last_frame: int

class IdentityRecovery:
    def __init__(self, max_gap_frames=35, max_distance_px=100, min_similarity=.75):
        self.max_gap=max_gap_frames
        self.max_distance=max_distance_px
        self.min_similarity=min_similarity
        self.raw_to_stable={}
        self.states={}
        self.next_id=1

    @staticmethod
    def _hist(frame,bbox):
        h,w=frame.shape[:2]
        x1,y1,x2,y2=[int(v) for v in bbox]
        x1=max(0,min(w-1,x1));x2=max(0,min(w,x2))
        y1=max(0,min(h-1,y1));y2=max(0,min(h,y2))
        if x2<=x1 or y2<=y1: return None
        crop=frame[y1:y2,x1:x2]
        if crop.size==0: return None
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,[16,16],[0,180,0,256])
        return cv2.normalize(hist,hist).flatten()

    def assign(self, frame, detections, frame_index):
        """Return stable ID for each detection, same order. Only re-associate recently lost IDs."""
        active_raw={d.tracker_id for d in detections}
        reserved=set()
        assignments=[]
        for det in detections:
            center=np.array([(det.bbox[0]+det.bbox[2])/2,(det.bbox[1]+det.bbox[3])/2],dtype=float)
            hist=self._hist(frame,det.bbox)
            sid=self.raw_to_stable.get(det.tracker_id)
            if sid is not None and sid in self.states and sid not in reserved:
                pass
            else:
                sid=None
                best_score=-1e9
                for candidate,old in self.states.items():
                    if candidate in reserved or old.raw_id in active_raw: continue
                    gap=frame_index-old.last_frame
                    if not 0 < gap <= self.max_gap: continue
                    predicted=old.center+old.velocity*min(gap,8)
                    dist=float(np.linalg.norm(center-predicted))
                    if dist > self.max_distance: continue
                    if hist is None or old.hist is None: continue
                    sim=float(cv2.compareHist(hist.astype('float32'),old.hist.astype('float32'),cv2.HISTCMP_CORREL))
                    if sim < self.min_similarity: continue
                    score=sim-0.35*dist/self.max_distance
                    if score>best_score: best_score=score;sid=candidate
                if sid is None:
                    sid=self.next_id;self.next_id+=1
                self.raw_to_stable[det.tracker_id]=sid
            old=self.states.get(sid)
            if old is not None:
                dt=max(1,frame_index-old.last_frame)
                velocity=.65*old.velocity+.35*(center-old.center)/dt
            else: velocity=np.zeros(2,dtype=float)
            self.states[sid]=IdentityState(sid,det.tracker_id,center,velocity,hist,frame_index)
            reserved.add(sid)
            assignments.append(sid)
        # Keep only recent states, and mappings for the retained states.
        self.states={k:v for k,v in self.states.items() if frame_index-v.last_frame<=self.max_gap}
        valid=set(self.states)
        self.raw_to_stable={k:v for k,v in self.raw_to_stable.items() if v in valid}
        return assignments
