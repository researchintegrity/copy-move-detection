import numpy as np
from sklearn.cluster import DBSCAN
from .utility import CMFD_PM_utily as utl
try:
    from scipy.ndimage import binary_fill_holes
except ImportError:
    from scipy.ndimage.morphology import binary_fill_holes

class Clusterer:
    def __init__(self, config):
        self.config = config

    def compute_local_dat(self, mpfY, mpfX):
        """
        Compute local affine transformation data.
        """
        dat = utl.MPF_DLFfit(mpfY, mpfX, radius=self.config['rd_fit'], mode='reflect')
        mask = np.all(np.isfinite(dat), -1)
        dat_cl = dat[mask, :]
        return dat_cl

    def cluster(self, img_shape, dat_cl):
        """
        Cluster the local data to find regions.
        """
        if len(dat_cl) == 0:
            return [], []

        dat_cl = np.asarray(dat_cl)
        fp = self.config['fact_pos']
        
        # Normalize
        # dat_cl columns: bx, ax, cx, px, by, ay, cy, py
        # indices:        0   1   2   3   4   5   6   7
        # px is 3, py is 7.
        
        # Normalization factors
        norm_factors = np.array([[1.0, 1.0, fp, fp, 1.0, 1.0, fp, fp]])
        dat_cl_norm = dat_cl / norm_factors
        
        clustering = DBSCAN(eps=0.5, min_samples=13).fit(dat_cl_norm).labels_
        num_clusters = np.max(clustering) + 1
        
        masks = np.zeros((num_clusters, img_shape[0], img_shape[1]), dtype=bool)
        center_clusters = np.zeros((num_clusters, 8))
        
        for i in range(num_clusters):
            pp = dat_cl[clustering == i]
            # pp[:, 7] is py, pp[:, 3] is px
            masks[i, np.uint32(pp[:, 7]), np.uint32(pp[:, 3])] = True
            masks[i] = utl.dilateDisk(masks[i], self.config['clustering_rd_dil'])
            masks[i] = binary_fill_holes(masks[i])
            center_clusters[i] = np.nanmean(pp, 0)
            
        return masks, center_clusters
