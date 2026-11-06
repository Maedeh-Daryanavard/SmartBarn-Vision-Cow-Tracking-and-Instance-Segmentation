"""Cow instance segmentation and tracked pixel masks (YOLO11-seg)."""
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import csv
import colorsys
import cv2
import numpy as np
from ultralytics import YOLO
from .preprocessing import AdaptivePreprocessor

@dataclass
class SegmentationConfig:
    video: str = "videos/cows.mp4"
    output_dir: str = "outputs/segmentation"
    weights: str = "yolo11n-seg.pt"
    tracker: str = "botsort.yaml"
    device: str = "cpu"
    conf: float = 0.15
    iou: float = 0.55
    imgsz: int = 960
    max_side: int = 960
    max_seconds: float = 10
    stride: int = 1
    enhancement: str = "auto"  # off, auto, always
    mask_threshold: float = 0.5
    alpha: float = 0.52
    save_individual_masks: bool = True
    save_instance_maps: bool = True
    h264: bool = True


def id_color(track_id):
    r,g,b = colorsys.hsv_to_rgb((int(track_id)*0.61803398875)%1, 0.82, 0.98)
    return tuple(int(v*255) for v in (b,g,r))


def exclusive_masks(masks, confidences, ids):
    """Resolve overlapping masks: higher-confidence instance owns overlapping pixels.
    Does not infer occluded pixels or guarantee biologically correct boundaries.
    Returns uint16 instance IDs; 0=background. IDs must fit 1..65535.
    """
    if len(masks) == 0:
        raise ValueError("Pass at least one mask")
    h,w=masks[0].shape
    label_map=np.zeros((h,w),dtype=np.uint16)
    # Low-confidence first; high-confidence predictions overwrite overlapping areas.
    for idx in np.argsort(np.asarray(confidences)):
        track_id=int(ids[idx])
        if not (1 <= track_id <= 65535):
            raise ValueError("Track ID exceeds uint16 PNG range")
        label_map[masks[idx].astype(bool)] = track_id
    return label_map


class CowInstanceSegmenter:
    def __init__(self, config=None):
        self.cfg=config or SegmentationConfig()
        if self.cfg.stride < 1: raise ValueError("stride must be >= 1")
        self.model=YOLO(self.cfg.weights)
        if self.model.task != "segment":
            raise ValueError("Use segmentation weights such as yolo11n-seg.pt")
        self.preprocessor=AdaptivePreprocessor(mode=self.cfg.enhancement)

    def predict_image(self, image, save_to=None):
        """Return (annotated BGR, uint16 instance map, records).
        Single-image IDs are temporary 1..N and are not stable across images.
        """
        if isinstance(image,(str,Path)):
            image=cv2.imread(str(image),cv2.IMREAD_COLOR)
            if image is None: raise ValueError("Could not read image")
        image=self._bgr(image)
        processed,mode=self.preprocessor.apply(image)
        result=self.model.predict(processed,classes=[19],conf=self.cfg.conf,
            iou=self.cfg.iou,imgsz=self.cfg.imgsz,device=self.cfg.device,retina_masks=True,verbose=False)[0]
        annotated,labels,records=self._render(image,result,0,0.0,mode,tracking=False)
        if save_to is not None:
            target=Path(save_to);target.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(target),annotated)
            cv2.imwrite(str(target.with_name(target.stem+"_ids.png")),labels)
        return annotated,labels,records

    @staticmethod
    def _bgr(frame):
        if frame.ndim==2:return cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        if frame.ndim==3 and frame.shape[2]==1:return cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        if frame.ndim==3 and frame.shape[2]==4:return cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
        return frame

    def _render(self,frame,result,frame_no,seconds,mode,tracking=True):
        h,w=frame.shape[:2]
        label_map=np.zeros((h,w),np.uint16)
        records=[]
        boxes=result.boxes
        if boxes is None or len(boxes)==0 or result.masks is None:
            return frame.copy(),label_map,records
        # Masks may be at model-input resolution. Resize with nearest neighbor.
        masks=result.masks.data.cpu().numpy()
        if len(masks)!=len(boxes):
            raise RuntimeError("Mask/box count mismatch")
        confidences=boxes.conf.cpu().numpy()
        coords=boxes.xyxy.cpu().numpy()
        if tracking and boxes.id is not None:
            ids=boxes.id.int().cpu().numpy()
        elif tracking:
            # Never fabricate persistent identities for unmatched detections.
            return frame.copy(),label_map,records
        else:
            ids=np.arange(1,len(boxes)+1,dtype=int)
        binary=[]
        for mask in masks:
            resized=cv2.resize(mask.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
            binary.append(resized>=self.cfg.mask_threshold)
        label_map=exclusive_masks(binary,confidences,ids)
        overlay=frame.copy()
        for tid in np.unique(label_map):
            if tid==0:continue
            overlay[label_map==tid]=id_color(tid)
        annotated=cv2.addWeighted(overlay,self.cfg.alpha,frame,1-self.cfg.alpha,0)
        for (x1,y1,x2,y2),tid,confidence in zip(coords,ids,confidences):
            tid=int(tid);pixel_count=int(np.count_nonzero(label_map==tid))
            if not pixel_count:continue
            mask8=np.uint8(label_map==tid)*255
            contours,_=cv2.findContours(mask8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated,contours,-1,id_color(tid),2)
            x,y=int(x1),int(y1)
            text=f"COW-{tid:04d} {confidence:.2f}"
            (tw,th),base=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
            x=max(0,min(x,w-tw-10));y=max(th+10,min(y-5,h-5))
            cv2.rectangle(annotated,(x,y-th-8),(x+tw+8,y+base+3),(255,255,255),-1)
            cv2.putText(annotated,text,(x+4,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(15,15,15),1,cv2.LINE_AA)
            records.append(dict(frame=frame_no,time_seconds=round(seconds,4),track_id=tid,
                confidence=float(confidence),pixels=pixel_count,
                x1=float(x1),y1=float(y1),x2=float(x2),y2=float(y2),enhancement=mode))
        return annotated,label_map,records

    def process_video(self):
        cfg=self.cfg
        source=Path(cfg.video)
        if not source.is_file():raise FileNotFoundError(f"Missing video: {source.resolve()}")
        out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(str(source))
        if not cap.isOpened():raise RuntimeError("Cannot open video")
        fps=cap.get(cv2.CAP_PROP_FPS) or 20.0
        ow=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));oh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale=min(1.0,cfg.max_side/max(ow,oh))
        w=max(2,int(ow*scale)//2*2);h=max(2,int(oh*scale)//2*2)
        video_path=out/"cow_instance_segmentation_mp4v.mp4"
        writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),fps/cfg.stride,(w,h))
        if not writer.isOpened():cap.release();raise RuntimeError("Video writer failed")
        if cfg.save_instance_maps:(out/"instance_maps").mkdir(exist_ok=True)
        if cfg.save_individual_masks:(out/"individual_masks").mkdir(exist_ok=True)
        records=[];frame_no=0;processed=0;max_frames=int(cfg.max_seconds*fps) if cfg.max_seconds else float("inf")
        try:
            while frame_no < max_frames:
                ok,frame=cap.read()
                if not ok:break
                current=frame_no;frame_no+=1
                if current%cfg.stride:continue
                frame=cv2.resize(self._bgr(frame),(w,h),interpolation=cv2.INTER_AREA)
                enhanced,mode=self.preprocessor.apply(frame)
                result=self.model.track(enhanced,persist=True,tracker=cfg.tracker,
                    classes=[19],conf=cfg.conf,iou=cfg.iou,imgsz=cfg.imgsz,
                    device=cfg.device,retina_masks=True,verbose=False)[0]
                annotated,labels,rows=self._render(frame,result,current,current/fps,mode)
                writer.write(annotated);records.extend(rows);processed+=1
                if cfg.save_instance_maps:
                    cv2.imwrite(str(out/"instance_maps"/f"frame_{current:06d}.png"),labels)
                if cfg.save_individual_masks:
                    for tid in np.unique(labels):
                        if tid==0:continue
                        cv2.imwrite(str(out/"individual_masks"/f"frame_{current:06d}_cow_{tid:04d}.png"),
                            np.uint8(labels==tid)*255)
                if processed%25==0:print(f"Processed {processed} frames; {len(rows)} tracked cows in latest frame")
        finally:
            cap.release();writer.release()
        csv_path=out/"instances.csv"
        columns=['frame','time_seconds','track_id','confidence','pixels','x1','y1','x2','y2','enhancement']
        with csv_path.open('w',newline='',encoding='utf8') as f:
            writer_csv=csv.DictWriter(f,fieldnames=columns);writer_csv.writeheader();writer_csv.writerows(records)
        final_path=video_path
        if cfg.h264:
            try:
                import imageio_ffmpeg,subprocess
                final_path=out/"cow_instance_segmentation_H264.mp4"
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error',
                    '-i',str(video_path),'-c:v','libx264','-pix_fmt','yuv420p',
                    '-crf','23','-movflags','+faststart',str(final_path)],check=True)
            except Exception as e:
                print('H.264 conversion unavailable; use MP4V output:',e)
                final_path=video_path
        return {'video':final_path,'raw_video':video_path,'csv':csv_path,
                'frames':processed,'detections':len(records),'unique_track_ids':len(set(r['track_id'] for r in records))}
