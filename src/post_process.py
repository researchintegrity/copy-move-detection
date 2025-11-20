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

    def process_double(self, mpfYA, mpfXA, mpfYB, mpfXB, input_maskA=None, input_maskB=None):
        """
        Process the MatchPatch fields for double image detection.
        """
        mpfYA = mpfYA.astype(np.float64); mpfXA = mpfXA.astype(np.float64)
        mpfYB = mpfYB.astype(np.float64); mpfXB = mpfXB.astype(np.float64)
        
        # Regularize offsets
        mpfYA, mpfXA = utl.MPFregularize(mpfYA, mpfXA, self.config['rd_median'])
        mpfYB, mpfXB = utl.MPFregularize(mpfYB, mpfXB, self.config['rd_median'])
        
        # DLF Error
        DLFerrA = utl.MPF_DLFerror(mpfYA, mpfXA, self.config['rd_dlf'])
        DLFscaleA = utl.MPF_DLFscale(mpfYA, mpfXA, self.config['rd_dlf'])
        DLFerrB = utl.MPF_DLFerror(mpfYB, mpfXB, self.config['rd_dlf'])
        DLFscaleB = utl.MPF_DLFscale(mpfYB, mpfXB, self.config['rd_dlf'])
        
        maskA = (DLFerrA <= self.config['th2_dlf']) & (DLFscaleA > self.config['th_scale'])
        maskB = (DLFerrB <= self.config['th2_dlf']) & (DLFscaleB > self.config['th_scale'])
        
        if input_maskA is not None:
            maskA = maskA & input_maskA
        if input_maskB is not None:
            maskB = maskB & input_maskB
            
        # Removal of small regions
        maskA = utl.removesmall(maskA, self.config['th_sizeA'], connectivity=8)
        maskB = utl.removesmall(maskB, self.config['th_sizeA'], connectivity=8)
        
        # Mirroring check (Cross-check)
        mskB_dil = utl.dilateDisk(maskB, self.config['rd_median'])
        mskA_dil = utl.dilateDisk(maskA, self.config['rd_median'])
        
        # mskAp: A points to B
        # maskBp: B is pointed to by A
        mskAp, maskBp = utl.MPFmirror(mpfYA.astype(np.int32), mpfXA.astype(np.int32), maskA, mskB_dil)
        
        # mskBp_rev: B points to A
        # maskAp_rev: A is pointed to by B
        mskBp_rev, maskAp_rev = utl.MPFmirror(mpfYB.astype(np.int32), mpfXB.astype(np.int32), maskB, mskA_dil)
        
        maskA = (mskAp == 1) | (maskAp_rev == 1)
        maskB = (mskBp_rev == 1) | (maskBp == 1)
        
        # Store mask before final dilation
        maskA_no_dil = maskA.copy()
        maskB_no_dil = maskB.copy()
        
        # Morphological operations
        maskA = utl.dilateDisk(maskA, self.config['rd_dil'])
        maskB = utl.dilateDisk(maskB, self.config['rd_dil'])
        maskA = binary_fill_holes(maskA)
        maskB = binary_fill_holes(maskB)
        
        return maskA, maskB, mpfYA, mpfXA, mpfYB, mpfXB, maskA_no_dil, maskB_no_dil
