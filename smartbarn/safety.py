"""Demonstration-only proximity alerts; NOT a safety-certified control system."""
import numpy as np

class RobotSafetySimulator:
    def __init__(self,path_normalized,radius_px=70,horizon_s=2.0):
        self.path=path_normalized
        self.radius=radius_px
        self.horizon=horizon_s
        self.previous={}

    def pixel_path(self,w,h):
        return [np.array([x*w,y*h],dtype=float) for x,y in self.path]

    @staticmethod
    def distance_to_polyline(point,polyline):
        d=float('inf')
        for a,b in zip(polyline[:-1],polyline[1:]):
            ab=b-a
            t=float(np.clip(np.dot(point-a,ab)/(np.dot(ab,ab)+1e-9),0,1))
            d=min(d,float(np.linalg.norm(point-(a+t*ab))))
        return d

    def update(self,cow_id,point,time_s,path):
        p=np.asarray(point,dtype=float)
        velocity=np.zeros(2)
        if cow_id in self.previous:
            last_t,last_p=self.previous[cow_id]
            dt=time_s-last_t
            if dt>0: velocity=(p-last_p)/dt
        self.previous[cow_id]=(time_s,p)
        dist=self.distance_to_polyline(p,path)
        future=p+velocity*self.horizon
        future_dist=self.distance_to_polyline(future,path)
        # Alerts indicate proximity to a STATIC hypothetical route, not collision probability.
        status='STOP' if dist<=self.radius else ('CAUTION' if future_dist<=self.radius else 'CLEAR')
        return dict(safety_status=status,distance_to_route_px=round(dist,1),predicted_distance_px=round(future_dist,1))
