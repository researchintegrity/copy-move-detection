import numpy as np
import sys
import os

# Try to import pm3D. It might be a .so file in the same directory.
try:
    from . import pm3D
except ImportError:
    try:
        import pm3D
    except ImportError:
        # If running from parent directory, we might need to add current dir to path
        sys.path.append(os.path.dirname(__file__))
        import pm3D

class Matcher:
    def __init__(self, config):
        self.config = config

    def match(self, feat):
        """
        Run PatchMatch on a single feature map (self-matching).
        """
        match_num_iter = self.config['match_num_iter']
        match_th_dist1 = self.config['match_th_dist1']
        match_num_tile = self.config['match_num_tile']
        match_diameter = self.config['match_diameter']

        # Normalize features to [0, 1]
        feat_min = np.min(feat)
        feat_max = np.max(feat)
        if feat_max > feat_min:
            feat = (feat - feat_min) / (feat_max - feat_min)
        
        # Reshape for pm3D: (H, W, 1, D)
        feat = np.reshape(feat, (feat.shape[0], feat.shape[1], 1, feat.shape[2]))
        
        # pm3D.pm3Dmod(diameter, 1, iterations, -th_dist1, 0, num_tile, source, target)
        cnn = pm3D.pm3Dmod(match_diameter, 1, match_num_iter,
                           -match_th_dist1, 0, match_num_tile, feat, feat)
                           
        if match_diameter > 1:
            mpfY = cnn[:-(match_diameter-1), :-(match_diameter-1), 0, 1].astype(np.int16)
            mpfX = cnn[:-(match_diameter-1), :-(match_diameter-1), 0, 0].astype(np.int16)
        else:
            mpfY = cnn[:, :, 0, 1].astype(np.int16)
            mpfX = cnn[:, :, 0, 0].astype(np.int16)
            
        return mpfY, mpfX

    def match_double(self, featA, featB):
        """
        Run PatchMatch between two feature maps.
        """
        # Use match parameters or defaults for double matching if they existed
        # For now reusing the main match parameters
        match_num_iter = self.config.get('match2_num_iter', 16)
        match_th_dist1 = self.config.get('match2_th_dist1', 0)
        match_num_tile = self.config.get('match2_num_tile', 1)
        match_diameter = self.config.get('match2_diameter', 1)

        # Normalize features together
        feat_min = min(np.min(featA), np.min(featB))
        feat_max = max(np.max(featA), np.max(featB))
        
        if feat_max > feat_min:
            featA = (featA - feat_min) / (feat_max - feat_min)
            featB = (featB - feat_min) / (feat_max - feat_min)
            
        featA = np.reshape(featA, (featA.shape[0], featA.shape[1], 1, featA.shape[2]))
        featB = np.reshape(featB, (featB.shape[0], featB.shape[1], 1, featB.shape[2]))

        cnn = pm3D.pm3Dmod(match_diameter, 1, match_num_iter, 
                           -match_th_dist1, 0, match_num_tile, featA, featB)
                           
        if match_diameter > 1:
            mpfY = cnn[:-(match_diameter-1), :-(match_diameter-1), 0, 1].astype(np.int16)
            mpfX = cnn[:-(match_diameter-1), :-(match_diameter-1), 0, 0].astype(np.int16)
        else:
            mpfY = cnn[:, :, 0, 1].astype(np.int16)
            mpfX = cnn[:, :, 0, 0].astype(np.int16)
            
        return mpfY, mpfX
