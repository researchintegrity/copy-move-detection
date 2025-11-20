import numpy as np
from .config import get_default_parameters
from .feature_extraction import FeatureExtractor
from .matching import Matcher
from .post_process import PostProcessor
from .clustering import Clusterer
from .analysis import Analyzer
from .visualization import Visualizer

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
        self.image = image
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

class CrossImageCopyDetector:
    def __init__(self, config=None):
        self.config = get_default_parameters()
        if config:
            self.config.update(config)
            
        self.feature_extractor = FeatureExtractor(self.config)
        self.matcher = Matcher(self.config)
        self.post_processor = PostProcessor(self.config)
        self.clusterer = Clusterer(self.config)
        self.visualizer = Visualizer()
        
        # Double image state
        self.imageA = None
        self.imageB = None
        self.featuresA = None
        self.featuresB = None
        self.padsizeA = None
        self.padsizeB = None
        self.mpfYA = None
        self.mpfXA = None
        self.mpfYB = None
        self.mpfXB = None
        self.maskA = None
        self.maskB = None
        self.mpfYA_reg = None
        self.mpfXA_reg = None
        self.mpfYB_reg = None
        self.mpfXB_reg = None
        self.maskA_no_dil = None
        self.maskB_no_dil = None
        self.clustersA = None
        self.clustersB = None
        self.dat_clA = None
        self.dat_clB = None

    def load_image_files(self, imageA_path, imageB_path):
        from .utility.utilityImage import imread2f
        self.imageA = imread2f(imageA_path)
        self.imageB = imread2f(imageB_path)
        return self.imageA, self.imageB

    def extract_features(self):
        if self.imageA is None or self.imageB is None:
            raise ValueError("Images not loaded")
            
        self.featuresA, self.padsizeA, self.diameter = self.feature_extractor.extract(self.imageA)
        self.featuresB, self.padsizeB, _ = self.feature_extractor.extract(self.imageB)
        return self.featuresA, self.featuresB

    def match_features(self):
        if self.featuresA is None or self.featuresB is None:
            raise ValueError("Features not extracted")
            
        # Match A -> B
        self.mpfYA, self.mpfXA = self.matcher.match_double(self.featuresA, self.featuresB)
        # Match B -> A
        self.mpfYB, self.mpfXB = self.matcher.match_double(self.featuresB, self.featuresA)
        
        return self.mpfYA, self.mpfXA, self.mpfYB, self.mpfXB

    def post_process(self):
        if self.mpfYA is None:
            raise ValueError("Matches not computed")
            
        self.maskA, self.maskB, self.mpfYA_reg, self.mpfXA_reg, self.mpfYB_reg, self.mpfXB_reg, self.maskA_no_dil, self.maskB_no_dil = \
            self.post_processor.process_double(self.mpfYA, self.mpfXA, self.mpfYB, self.mpfXB)
            
        return self.maskA, self.maskB

    def cluster_detections(self):
        if self.mpfYA_reg is None:
            raise ValueError("Post-processing not done")
            
        # Cluster A
        mpfYA_masked = self.mpfYA_reg.copy()
        mpfXA_masked = self.mpfXA_reg.copy()
        maskA_bool = self.maskA_no_dil.astype(bool)
        
        mpfYA_masked[~maskA_bool] = np.nan
        mpfXA_masked[~maskA_bool] = np.nan
        
        self.dat_clA = self.clusterer.compute_local_dat(mpfYA_masked, mpfXA_masked)
        self.clustersA, _ = self.clusterer.cluster(self.maskA.shape, self.dat_clA)
        
        # Cluster B
        mpfYB_masked = self.mpfYB_reg.copy()
        mpfXB_masked = self.mpfXB_reg.copy()
        maskB_bool = self.maskB_no_dil.astype(bool)
        
        mpfYB_masked[~maskB_bool] = np.nan
        mpfXB_masked[~maskB_bool] = np.nan
        
        self.dat_clB = self.clusterer.compute_local_dat(mpfYB_masked, mpfXB_masked)
        self.clustersB, _ = self.clusterer.cluster(self.maskB.shape, self.dat_clB)
        
        return self.clustersA, self.clustersB

    def get_final_masks(self):
        """
        Return the masks padded to the original image size.
        """
        if self.maskA is None or self.maskB is None:
            return None, None
        
        maskA_final = self.maskA
        maskB_final = self.maskB
        
        if self.padsizeA:
            maskA_final = np.pad(self.maskA, self.padsizeA, 'constant', constant_values=False)
        if self.padsizeB:
            maskB_final = np.pad(self.maskB, self.padsizeB, 'constant', constant_values=False)
            
        return maskA_final, maskB_final

    def get_final_clusters(self):
        """
        Return the cluster masks padded to the original image size.
        """
        if self.clustersA is None or self.clustersB is None:
            return None, None
            
        clustersA_final = self.clustersA
        clustersB_final = self.clustersB
        
        if self.padsizeA:
            clustersA_final = []
            for cluster_mask in self.clustersA:
                clustersA_final.append(np.pad(cluster_mask, self.padsizeA, 'constant', constant_values=False))
            clustersA_final = np.array(clustersA_final)
            
        if self.padsizeB:
            clustersB_final = []
            for cluster_mask in self.clustersB:
                clustersB_final.append(np.pad(cluster_mask, self.padsizeB, 'constant', constant_values=False))
            clustersB_final = np.array(clustersB_final)
            
        return clustersA_final, clustersB_final

    def run(self, imageA, imageB):
        self.imageA = imageA
        self.imageB = imageB
        self.extract_features()
        self.match_features()
        self.post_process()
        self.cluster_detections()
        
        maskA, maskB = self.get_final_masks()
        clustersA, clustersB = self.get_final_clusters()
        
        return maskA, maskB, clustersA, clustersB

    def visualize_matches(self, max_lines=500, figsize=(18, 6)):
        """
        Visualize the matches for double image detection.
        """
        if self.imageA is None or self.imageB is None or self.maskA is None:
            print("Data missing for visualization")
            return
            
        mpfYA = self.mpfYA_reg if self.mpfYA_reg is not None else self.mpfYA
        mpfXA = self.mpfXA_reg if self.mpfXA_reg is not None else self.mpfXA
        
        self.visualizer.plot_matches_double(self.imageA, self.imageB, self.maskA, mpfYA, mpfXA, 
                                            self.padsizeA, self.padsizeB, self.clustersA, max_lines, figsize)

    def visualize_clusters(self, figsize=(18, 6)):
        """
        Visualize the clusters for double image detection.
        """
        if self.imageA is None or self.imageB is None or self.clustersA is None or self.clustersB is None:
            print("Data missing for visualization")
            return
            
        # Use regularized offsets
        mpfYA = self.mpfYA_reg if self.mpfYA_reg is not None else self.mpfYA
        mpfXA = self.mpfXA_reg if self.mpfXA_reg is not None else self.mpfXA
        mpfYB = self.mpfYB_reg if self.mpfYB_reg is not None else self.mpfYB
        mpfXB = self.mpfXB_reg if self.mpfXB_reg is not None else self.mpfXB
        
        self.visualizer.plot_clusters_double(self.imageA, self.imageB, self.clustersA, self.clustersB,
                                             mpfYA, mpfXA, mpfYB, mpfXB,
                                             self.padsizeA, self.padsizeB, figsize)
