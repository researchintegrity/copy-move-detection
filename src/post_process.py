import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from .utility import CMFD_PM_utily as utl

class PostProcessor:
    def __init__(self, config):
        self.config = config

    def process(self, mpfY, mpfX, input_mask=None, remove_msk=None):
        """
        Process the MatchPatch fields to generate a detection mask.
        Returns:
            mask: The final detection mask.
            mpfY: Regularized Y offsets.
            mpfX: Regularized X offsets.
            DLFerr: Dense Linear Fitting error.
            DLFscale: Dense Linear Fitting scale.
        """
        mpfY = mpfY.astype(np.float64)
        mpfX = mpfX.astype(np.float64)
        
        # Regularize offsets
        mpfY, mpfX = utl.MPFregularize(mpfY, mpfX, self.config['rd_median'])
        
        # DLF Error
        DLFerr = utl.MPF_DLFerror(mpfY, mpfX, self.config['rd_dlf'])
        DLFscale = utl.MPF_DLFscale(mpfY, mpfX, self.config['rd_dlf'])
        
        mask = (DLFerr <= self.config['th2_dlf']) & (DLFscale > self.config['th_scale'])
        
        if input_mask is not None:
            mask = mask & input_mask
            
        # Removal of too close duplicate candidates
        dist2 = utl.MPFspacedist2(mpfY, mpfX)
        mask = mask & (dist2 >= self.config['th2_dist2'])
        
        # Removal of small regions
        mask = utl.removesmall(mask, self.config['th_sizeA'], connectivity=8)
        mask = utl.MPFdual(mpfY, mpfX, mask) # Mirroring
        mask = utl.removesmall(mask, self.config['th_sizeB'], connectivity=8)
        
        if remove_msk is not None:
            mask = mask & remove_msk
            
        # Removal regions without mirror
        msk = utl.dilateDisk(mask, self.config['rd_median'])
        msk, mask = utl.MPFmirror(mpfY.astype(np.int32), mpfX.astype(np.int32), mask, msk)
        mask = (mask == 1) | (msk == 1)
        
        # Store mask before final dilation for clustering purposes
        mask_no_dil = mask.copy()
        
        # Morphological operations
        mask = utl.dilateDisk(mask, self.config['rd_dil'])
        mask = binary_fill_holes(mask)
        
        return mask, mpfY, mpfX, DLFerr, DLFscale, mask_no_dil
