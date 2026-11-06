from collections import defaultdict,deque
import cv2
import numpy as np

class Visualizer:
    def __init__(self,history_length=45):
        self.history=defaultdict(lambda:deque(maxlen=history_length))

    @staticmethod
    def color(cow_id):
        hue=int((cow_id*43)%180)
        pixel=np.uint8([[[hue,185,230]]])
        return tuple(int(x) for x in cv2.cvtColor(pixel,cv2.COLOR_HSV2BGR)[0,0])

    def draw_background(self,frame,polygons,route):
        for name,poly in polygons.items():
            cv2.polylines(frame,[poly],True,(80,200,240),2)
            x,y=poly[0];cv2.putText(frame,name,(int(x),max(18,int(y)-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(80,200,240),2)
        if route:
            pts=np.array(route,dtype=np.int32)
            cv2.polylines(frame,[pts],False,(0,180,255),3)
            cv2.putText(frame,'SIMULATED ROBOT ROUTE',tuple(pts[0].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,180,255),1)

    def draw_cow(self,frame,cow_id,bbox,confidence,anchor,activity,zones,safety):
        h,w=frame.shape[:2]
        x1,y1,x2,y2=[int(v) for v in bbox]
        col=self.color(cow_id)
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
        label=f'COW-{cow_id:04d}'
        label_w=112;label_h=29
        lx=max(0,min(w-label_w,x1));ly=max(0,y1-label_h-5)
        cv2.rectangle(frame,(lx,ly),(lx+label_w,ly+label_h),(245,245,245),-1)
        cv2.rectangle(frame,(lx+label_w-24,ly),(lx+label_w,ly+label_h),col,-1)
        cv2.putText(frame,label,(lx+5,ly+20),cv2.FONT_HERSHEY_SIMPLEX,.47,(25,25,25),2,cv2.LINE_AA)
        self.history[cow_id].append(tuple(map(int,anchor)))
        pts=np.array(self.history[cow_id],dtype=np.int32)
        if len(pts)>1:cv2.polylines(frame,[pts],False,col,2)
        cv2.circle(frame,tuple(map(int,anchor)),4,col,-1)
        info=f'{activity} | {safety} | {confidence:.2f}'
        cv2.putText(frame,info,(max(0,x1),min(h-6,y2+17)),cv2.FONT_HERSHEY_SIMPLEX,.39,col,1,cv2.LINE_AA)

    @staticmethod
    def header(frame,time_s,count,mode):
        cv2.rectangle(frame,(0,0),(frame.shape[1],55),(20,20,20),-1)
        cv2.putText(frame,f'SMARTBARN | cows tracked: {count} | t={time_s:.1f}s',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        cv2.putText(frame,f'Preprocessing: {mode} | IDs are tracker estimates',(10,44),cv2.FONT_HERSHEY_SIMPLEX,.43,(195,215,230),1)
