import cv2
import numpy as np

class AdaptivePreprocessor:
    def __init__(self, mode='auto', clip=2.0, grid=(8,8)):
        if mode not in ('auto','always','off'):
            raise ValueError('mode must be auto, always, or off')
        self.mode=mode
        self.clahe=cv2.createCLAHE(clipLimit=clip,tileGridSize=grid)

    def apply(self, frame):
        if frame.ndim == 2:
            frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        sample=frame[::8,::8].astype(np.float32)
        channel_delta=float(np.mean(np.max(sample,axis=2)-np.min(sample,axis=2)))
        almost_gray=channel_delta < 8
        low_contrast=float(np.std(gray)) < 42
        if self.mode=='off' or (self.mode=='auto' and not (almost_gray or low_contrast)):
            return frame,'original'
        if almost_gray:
            return cv2.cvtColor(self.clahe.apply(gray),cv2.COLOR_GRAY2BGR),'clahe_gray'
        lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab)
        enhanced=cv2.cvtColor(cv2.merge([self.clahe.apply(l),a,b]),cv2.COLOR_LAB2BGR)
        return enhanced,'clahe_color'
