import numpy as np
from .config import get_default_parameters
from .feature_extraction import FeatureExtractor
from .matching import Matcher
from .post_process import PostProcessor
from .clustering import Clusterer
from .analysis import Analyzer
from .visualization import Visualizer
from .utility.utilityImage import img2grayf

class CopyMoveDetector:
    def __init__(self, config=None):
        self.config = get_default_parameters()
        if config:
            self.config.update(config)
            
        self.feature_extractor = FeatureExtractor(self.config)
        self.matcher = Matcher(self.config)
        self.post_processor = PostProcessor(self.config)
        self.clusterer = Clusterer(self.config)
        self.analyzer = Analyzer()
        self.visualizer = Visualizer()
        
        # State storage for debugging
        self.image = None
        self.features = None
        self.padsize = None
        self.diameter = None
        self.mpfY = None
        self.mpfX = None
        self.mask = None
        self.mpfY_reg = None
        self.mpfX_reg = None
        self.DLFerr = None
        self.DLFscale = None
        self.clusters = None
        self.cluster_centers = None
        self.dat_cl = None
        self.analysis_results = None
        self.mask_no_dil = None

    def load_image(self, image):
        self.image = image
        return self.image

    def load_image_file(self, image_path):
        """
        Load image from file using the same method as src2 (PIL based).
        This ensures consistent color space (RGB) and scaling.
        """
        from .utility.utilityImage import imread2f
        self.image = imread2f(image_path)
        return self.image

    def extract_features(self, image=None):
        if image is not None:
            self.image = image
        if self.image is None:
            raise ValueError("No image loaded")
            
        self.features, self.padsize, self.diameter = self.feature_extractor.extract(self.image)
        return self.features

    def match_features(self, features=None):
        if features is not None:
            self.features = features
        if self.features is None:
            raise ValueError("No features extracted")
            
        self.mpfY, self.mpfX = self.matcher.match(self.features)
        return self.mpfY, self.mpfX

    def post_process(self, mpfY=None, mpfX=None):
        if mpfY is not None:
            self.mpfY = mpfY
        if mpfX is not None:
            self.mpfX = mpfX
            
        if self.mpfY is None or self.mpfX is None:
            raise ValueError("No matches found")
            
        # Note: PostProcessor.process returns mask, mpfY, mpfX, DLFerr, DLFscale, mask_no_dil
        # But mpfY and mpfX are regularized versions.
        self.mask, self.mpfY_reg, self.mpfX_reg, self.DLFerr, self.DLFscale, self.mask_no_dil = \
            self.post_processor.process(self.mpfY, self.mpfX)
            
        return self.mask

    def cluster_detections(self):
        if self.mpfY_reg is None:
            raise ValueError("Post-processing not done")
            
        # Apply mask to regularized offsets for clustering
        # Use mask_no_dil to match src2 behavior (exclude dilated regions from clustering data)
        mpfY_masked = self.mpfY_reg.copy()
        mpfX_masked = self.mpfX_reg.copy()
        
        # Ensure mask is boolean
        mask_bool = self.mask_no_dil.astype(bool)
        
        mpfY_masked[~mask_bool] = np.nan
        mpfX_masked[~mask_bool] = np.nan
        
        self.dat_cl = self.clusterer.compute_local_dat(mpfY_masked, mpfX_masked)
        
        # We need to pass the original image shape, but mask is smaller if valid padding was used.
        # The clusters should probably be mapped back to the original image size?
        # Or we cluster on the feature map size and then pad the result.
        # The clusterer returns masks of size passed in.
        
        self.clusters, self.cluster_centers = self.clusterer.cluster(self.mask.shape, self.dat_cl)
        
        return self.clusters

    def get_final_mask(self):
        """
        Return the mask padded to the original image size.
        """
        if self.mask is None:
            return None
        
        if self.padsize:
            return np.pad(self.mask, self.padsize, 'constant', constant_values=False)
        return self.mask

    def get_final_clusters(self):
        """
        Return the cluster masks padded to the original image size.
        """
        if self.clusters is None:
            return None
            
        if self.padsize:
            padded_clusters = []
            for cluster_mask in self.clusters:
                padded_clusters.append(np.pad(cluster_mask, self.padsize, 'constant', constant_values=False))
            return np.array(padded_clusters)
        return self.clusters

    def analyze_detections(self):
        """
        Analyze the clusters to extract geometric information.
        """
        if self.clusters is None or self.cluster_centers is None:
            raise ValueError("Clustering not done")
            
        self.analysis_results = self.analyzer.analyze_clusters(self.clusters, self.cluster_centers)
        return self.analysis_results

    def run(self, image):
        self.load_image(image)
        self.extract_features()
        self.match_features()
        self.post_process()
        self.cluster_detections()
        self.analyze_detections()
        return self.get_final_mask(), self.get_final_clusters()

    def visualize_matches(self, max_lines=500, figsize=(12, 12)):
        """
        Visualize the matches.
        """
        if self.image is None or self.mask is None:
            print("Data missing for visualization")
            return
            
        # Use regularized offsets if available, otherwise raw offsets
        mpfY = self.mpfY_reg if self.mpfY_reg is not None else self.mpfY
        mpfX = self.mpfX_reg if self.mpfX_reg is not None else self.mpfX
        
        self.visualizer.plot_matches(self.image, self.mask, mpfY, mpfX, self.padsize, self.clusters, max_lines, figsize)

    def visualize_clusters(self, figsize=(12, 12)):
        """
        Visualize the clusters.
        """
        if self.image is None or self.clusters is None:
            print("Data missing for visualization")
            return
            
        # Use regularized offsets if available
        mpfY = self.mpfY_reg if self.mpfY_reg is not None else self.mpfY
        mpfX = self.mpfX_reg if self.mpfX_reg is not None else self.mpfX

        self.visualizer.plot_clusters(self.image, self.clusters, self.padsize, mpfY, mpfX, figsize)
