import numpy as np
import cv2
from .utils import cleanImage, add_alpha_channel

threshold_modes = ['global', 'adaptive']

class ThresholdImageMaker():
    def __init__(self, threshold=0):
        self.threshold = threshold

    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def make_binary_image(self, im_path, to_path, threshold_mode='adaptive', clean_image=False, transparent_background=False):
        img = cv2.imread(im_path,0)
        
        assert threshold_mode in threshold_modes, "Invalid value of threshold_mode."
        if transparent_background:
            assert to_path.split('.')[-1] == 'png', "If transparent_background=True, to_path must be PNG."

        if threshold_mode == 'global':
            _, thresh = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
        elif threshold_mode == 'adaptive':
            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        if clean_image:
            thresh = cleanImage(thresh)

        if transparent_background:
            thresh = add_alpha_channel(np.stack([thresh]*3, axis=-1))

        cv2.imwrite(to_path, thresh)