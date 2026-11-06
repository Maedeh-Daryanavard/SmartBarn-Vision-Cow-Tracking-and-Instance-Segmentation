import cv2
import numpy as np

class ZoneMonitor:
    def __init__(self, zones):
        self.zones=zones
        self.inside_since={}
        self.events=[]
        self.durations={}

    def polygons(self,width,height):
        return {name:np.array([(int(x*width),int(y*height)) for x,y in coords],dtype=np.int32) for name,coords in self.zones.items()}

    def update(self, cow_id, point, time_s, polygons):
        present=[]
        for name,poly in polygons.items():
            key=(cow_id,name)
            inside=cv2.pointPolygonTest(poly,(float(point[0]),float(point[1])),False)>=0
            if inside:
                present.append(name)
                if key not in self.inside_since:
                    self.inside_since[key]=time_s
                    self.events.append(dict(time_s=time_s,cow_id=cow_id,zone=name,event='enter'))
            elif key in self.inside_since:
                self._close(key,time_s)
        return present

    def _close(self,key,time_s):
        since=self.inside_since.pop(key)
        self.durations[key]=self.durations.get(key,0.)+max(0.,time_s-since)
        self.events.append(dict(time_s=time_s,cow_id=key[0],zone=key[1],event='exit'))

    def finalize(self,time_s):
        for key in list(self.inside_since): self._close(key,time_s)
