import numpy as np
from time import time
from .utility import CHF
from .utility.utilityImage import img2grayf

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.diameter_feat = [12, 12, 12, 12, 24]
        self.diameter = self.diameter_feat[self.config['type_feat'] - 1]
        self.bfdata = self._generate_filters()

    def _generate_filters(self):
        type_feat = self.config['type_feat']
        diameter = self.diameter
        
        if type_feat == 1:
            return CHF.ZM_bf(diameter, self.config['ZM_order'])
        elif type_feat == 2:
            return CHF.ZMp_bf(diameter, self.config['ZM_order'], self.config['radiusNum'], self.config['anglesNum'])
        elif type_feat == 3:
            PCT_NM = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(2,0),(2,1),(3,0)]
            return CHF.PCT_bf(diameter, PCT_NM)
        elif type_feat == 4:
            PCT_NM = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(2,0),(2,1),(3,0)]
            return CHF.PCTp_bf(diameter, PCT_NM, self.config['radiusNum'], self.config['anglesNum'])
        elif type_feat == 5:
            FMT_N = range(-2, 3)
            FMT_M = range(5)
            return CHF.FMTpl_bf(diameter, FMT_M, self.config['radiusNum'], self.config['anglesNum'], FMT_N, self.config['radiusMin'])
        else:
            raise ValueError(f"Unknown feature type: {type_feat}")

    def extract(self, img):
        """
        Extract features from the image.
        Returns:
            feat: The extracted features.
            padsize: The padding size used.
            diameter: The diameter of the features.
        """
        img = img2grayf(img)
        
        raggioU = int(np.ceil((self.diameter - 1.0) / 2.0))
        raggioL = int(np.floor((self.diameter - 1.0) / 2.0))
        padsize = ((raggioU, raggioL), (raggioU, raggioL))
        
        if self.config['pad_img']:
            img = np.pad(img.copy(), padsize, mode='edge')
            padsize = ((0, 0), (0, 0))
            
        feat = np.abs(CHF.FiltersBank_FFT(img, self.bfdata, mode='valid'))
        return feat, padsize, self.diameter
