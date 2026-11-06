from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from .config import Config
from .preprocessing import AdaptivePreprocessor
from .detection import CowDetectorTracker
from .identity import IdentityRecovery
from .zones import ZoneMonitor
from .behavior import ActivityAnalyzer
from .safety import RobotSafetySimulator
from .visualization import Visualizer

class SmartBarnPipeline:
    def __init__(self,config:Config):
        self.cfg=config

    def run(self):
        c=self.cfg
        src=Path(c.video)
        if not src.exists(): raise FileNotFoundError(f'Missing input video: {src.resolve()}')
        out=Path(c.output_dir);out.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(str(src))
        if not cap.isOpened(): raise RuntimeError(f'Unable to open {src}')
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if fps<=0: fps=20.
        sw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));sh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if sw<=0 or sh<=0: cap.release();raise RuntimeError('Invalid video dimensions')
        scale=min(1.,c.max_side/max(sw,sh))
        w=max(2,int(sw*scale)//2*2);h=max(2,int(sh*scale)//2*2)
        if c.stride<1: cap.release();raise ValueError('stride must be >=1')
        output_video=out/'annotated_mp4v.mp4'
        writer=cv2.VideoWriter(str(output_video),cv2.VideoWriter_fourcc(*'mp4v'),fps/c.stride,(w,h))
        if not writer.isOpened(): cap.release();raise RuntimeError('Video writer failed')
        pre=AdaptivePreprocessor(c.enhancement)
        detector=CowDetectorTracker(c.weights,c.tracker,c.device,c.conf,c.imgsz,c.cow_class)
        recovery=IdentityRecovery(max_distance_px=max(60,int(w*.12))) if c.use_identity_recovery else None
        zones=ZoneMonitor(c.zones) if c.use_zones else None
        polys=zones.polygons(w,h) if zones else {}
        activity=ActivityAnalyzer(fps/c.stride,posture_model_path=c.posture_model_path) if c.use_behavior else None
        safety=RobotSafetySimulator(c.robot_path,c.safety_radius_px,c.warning_horizon_s) if c.use_safety else None
        route=safety.pixel_path(w,h) if safety else []
        viz=Visualizer()
        records=[];frame_index=0;processed=0;last_time=0.;start=time.perf_counter()
        max_frames=int(c.max_seconds*fps) if c.max_seconds>0 else float('inf')
        try:
            while frame_index<max_frames:
                ok,frame=cap.read()
                if not ok: break
                fi=frame_index;frame_index+=1
                if fi%c.stride: continue
                t=fi/fps;last_time=t
                frame=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
                enhanced,mode=pre.apply(frame)
                detections=detector.process(enhanced)
                ids=recovery.assign(enhanced,detections,fi) if recovery else [d.tracker_id for d in detections]
                annotated=frame.copy()
                viz.draw_background(annotated,polys,route)
                for det,sid in zip(detections,ids):
                    x1,y1,x2,y2=det.bbox
                    center=((x1+x2)/2,(y1+y2)/2)
                    # Center is used consistently for zone, motion, and safety estimates.
                    anchor=center
                    active_zones=zones.update(sid,anchor,t,polys) if zones else []
                    act=activity.update(sid,center,t,enhanced,det.bbox) if activity else dict(activity='unknown',motion_px_s=0.,posture='unknown')
                    safe=safety.update(sid,anchor,t,route) if safety else dict(safety_status='NOT_EVALUATED',distance_to_route_px=np.nan,predicted_distance_px=np.nan)
                    viz.draw_cow(annotated,sid,det.bbox,det.confidence,anchor,act['activity'],active_zones,safe['safety_status'])
                    records.append(dict(frame=fi,time_s=t,raw_tracker_id=det.tracker_id,cow_id=sid,confidence=det.confidence,x1=x1,y1=y1,x2=x2,y2=y2,center_x=center[0],center_y=center[1],zones=';'.join(active_zones),preprocessing=mode,**act,**safe))
                viz.header(annotated,t,len(detections),mode)
                writer.write(annotated)
                processed+=1
                if processed%50==0:print(f'Processed {processed} frames')
        finally:
            cap.release();writer.release()
        if zones:zones.finalize(last_time)
        df=pd.DataFrame(records)
        df.to_csv(out/'detections_tracks.csv',index=False)
        if zones:
            pd.DataFrame(zones.events,columns=['time_s','cow_id','zone','event']).to_csv(out/'zone_events.csv',index=False)
            pd.DataFrame([dict(cow_id=k[0],zone=k[1],seconds=v) for k,v in zones.durations.items()],columns=['cow_id','zone','seconds']).to_csv(out/'zone_durations.csv',index=False)
        if not df.empty:
            summary=df.groupby('cow_id').agg(observations=('frame','size'),first_seen_s=('time_s','min'),last_seen_s=('time_s','max'),mean_confidence=('confidence','mean')).reset_index()
        else:summary=pd.DataFrame(columns=['cow_id','observations','first_seen_s','last_seen_s','mean_confidence'])
        summary.to_csv(out/'track_summary.csv',index=False)
        elapsed=time.perf_counter()-start
        print(f'Finished: {processed} frames, {len(summary)} estimated IDs, {processed/max(elapsed,.001):.2f} processing FPS')
        print('Video:',output_video.resolve())
        return dict(video=output_video,records=df,summary=summary,output_dir=out,processed_frames=processed)
