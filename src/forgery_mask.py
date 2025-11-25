import argparse
import os
import sys
import numpy as np
import logging
import multiprocessing
import time
import cv2
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("forgery_mask")

try:
    from copy_move_detection.detector import CopyMoveDetector, CrossImageCopyDetector
    from copy_move_detection.utility.utilityImage import imread2f
except ImportError:
    # Ensure we can import from src locally
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.detector import CopyMoveDetector, CrossImageCopyDetector
    from src.utility.utilityImage import imread2f

def compute_forgery_groups(clusters, mpfY, mpfX, padsize, image_shape):
    """
    Group clusters based on their matching relationship and return a multi-channel mask.
    Each channel corresponds to a group of matched clusters.
    """
    if clusters is None or len(clusters) == 0:
        return np.zeros((0, image_shape[0], image_shape[1]), dtype=np.uint8)

    n_clusters = len(clusters)
    source_masks = []
    target_masks = []

    # 1. Compute Source and Target Masks for each cluster
    for i, src_mask in enumerate(clusters):
        source_masks.append(src_mask)
        
        # Compute Target Mask
        dst_mask = np.zeros_like(src_mask)
        
        y_idxs, x_idxs = np.where(src_mask)
        if len(y_idxs) > 0:
            # Use offsets to find target coordinates
            val_ys = mpfY[y_idxs, x_idxs]
            val_xs = mpfX[y_idxs, x_idxs]
            
            valid = ~np.isnan(val_ys) & ~np.isnan(val_xs)
            tys = np.round(val_ys[valid]).astype(int)
            txs = np.round(val_xs[valid]).astype(int)
            
            # Filter bounds
            H, W = dst_mask.shape
            in_bounds = (tys >= 0) & (tys < H) & (txs >= 0) & (txs < W)
            tys = tys[in_bounds]
            txs = txs[in_bounds]
            
            dst_mask[tys, txs] = True
            
            # Post-process dst_mask to remove outliers (same as visualization.py)
            dst_mask = cv2.morphologyEx(dst_mask.astype(np.uint8), cv2.MORPH_OPEN, 
                                np.ones((5, 5), np.uint8)).astype(bool)
                
        target_masks.append(dst_mask)

    # 2. Build Adjacency Matrix for Clusters
    adj_matrix = np.zeros((n_clusters, n_clusters), dtype=int)

    for i in range(n_clusters):
        # Region i is the union of its source and target
        region_i = source_masks[i] | target_masks[i]
        
        for j in range(i + 1, n_clusters):
            region_j = source_masks[j] | target_masks[j]
            
            # Check overlap
            overlap_count = np.logical_and(region_i, region_j).sum()
            if overlap_count > 5: 
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    # 3. Find Connected Components
    if n_clusters > 0:
        n_components, labels = connected_components(csr_matrix(adj_matrix), directed=False)
    else:
        n_components = 0
        labels = []

    # 4. Create Multi-channel Mask
    # Output shape: (n_components, H_img, W_img)
    final_masks = []

    for group_id in range(n_components):
        # Find all clusters in this group
        cluster_indices = np.where(labels == group_id)[0]
        
        # Combine all masks in this group
        group_mask = np.zeros_like(source_masks[0])
        for idx in cluster_indices:
            group_mask |= source_masks[idx]
            group_mask |= target_masks[idx]
            
        # Pad to image size
        if padsize:
            group_mask_padded = np.pad(group_mask, padsize, 'constant', constant_values=False)
        else:
            group_mask_padded = group_mask
        
        final_masks.append(group_mask_padded)
    
    if not final_masks:
        return np.zeros((0, image_shape[0], image_shape[1]), dtype=np.uint8)
        
    return np.array(final_masks, dtype=np.uint8)

def save_forgery_mask(detector, output_path):
    """
    Save the forgery mask as a .npy file.
    Uses the detector state to compute grouped clusters.
    """
    # Determine image shape
    if detector.image is not None:
        image_shape = detector.image.shape[:2]
    elif detector.mask is not None:
        # Estimate from mask and padsize
        h, w = detector.mask.shape
        if detector.padsize:
            h += detector.padsize[0][0] + detector.padsize[0][1]
            w += detector.padsize[1][0] + detector.padsize[1][1]
        image_shape = (h, w)
    else:
        logger.error("Cannot determine image shape for mask generation.")
        return

    clusters = detector.clusters if detector.clusters is not None else []
    
    if not clusters:
        logger.warning("No clusters found. Saving empty mask.")
        grouped_masks = np.zeros((0, image_shape[0], image_shape[1]), dtype=np.uint8)
    else:
        # Use regularized offsets if available
        mpfY = detector.mpfY_reg if detector.mpfY_reg is not None else detector.mpfY
        mpfX = detector.mpfX_reg if detector.mpfX_reg is not None else detector.mpfX
        
        if mpfY is None or mpfX is None:
            logger.warning("No offsets found in detector but clusters exist. Saving empty mask.")
            grouped_masks = np.zeros((0, image_shape[0], image_shape[1]), dtype=np.uint8)
        else:
            grouped_masks = compute_forgery_groups(clusters, mpfY, mpfX, detector.padsize, image_shape)
    
    # Save
    try:
        np.save(output_path, grouped_masks)
        logger.info(f"Saved forgery mask to {output_path} with shape {grouped_masks.shape}")
    except Exception as e:
        logger.error(f"Failed to save forgery mask to {output_path}: {e}")

def get_feature_name(type_feat):
    mapping = {
        1: 'ZM-cart',
        2: 'ZM-polar',
        3: 'PCT-cart',
        4: 'PCT-polar',
        5: 'FMT'
    }
    return mapping.get(type_feat, f"Unknown ({type_feat})")

def log_configuration(config):
    feat_name = get_feature_name(config.get('type_feat', 2))
    clustering_algo = config.get('clustering_algorithm', 'dbscan')
    logger.info("Configuration:")
    logger.info(f"  - Feature Extraction: {feat_name}")
    logger.info(f"  - Clustering Algorithm: {clustering_algo}")
    
    if clustering_algo == 'dbscan':
        logger.info(f"    - eps: {config.get('clustering_eps', 0.5)}")
        logger.info(f"    - min_samples: {config.get('clustering_min_samples', 13)}")
    elif clustering_algo == 'hdbscan':
        logger.info(f"    - min_cluster_size: {config.get('clustering_min_cluster_size', 13)}")
        logger.info(f"    - min_samples: {config.get('clustering_min_samples', 'None')}")
    elif clustering_algo == 'optics':
        logger.info(f"    - min_samples: {config.get('clustering_min_samples', 13)}")
        logger.info(f"    - max_eps: {config.get('clustering_max_eps', 'inf')}")
        logger.info(f"    - xi: {config.get('clustering_xi', 0.05)}")
    elif clustering_algo == 'agglomerative':
        logger.info(f"    - distance_threshold: {config.get('clustering_distance_threshold', 0.5)}")
    elif clustering_algo == 'meanshift':
        logger.info(f"    - bandwidth_quantile: {config.get('clustering_bandwidth_quantile', 0.2)}")

def status_logger(stage, progress):
    logger.info(f"Detector Status: {stage} - Progress: {progress*100:.1f}%")

def _run_single_detection_worker(image_path, output_dir, config):
    logger.info(f"Running Single Image Detection on {image_path}")
    log_configuration(config)
    
    detector = CopyMoveDetector(config)
    try:
        img = imread2f(image_path)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return

    try:
        logger.info("Starting detection pipeline...")
        start_time = time.time()
        mask, clusters = detector.run(img, status_callback=status_logger)
        elapsed_time = time.time() - start_time
        logger.info(f"Detection completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return

    # Save Forgery Mask .npy
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    npy_path = os.path.join(output_dir, f"{base_name}_forgery_mask.npy")
    save_forgery_mask(detector, npy_path)

def run_single_detection(image_path, output_dir, config, timeout=600):
    p = multiprocessing.Process(target=_run_single_detection_worker, args=(image_path, output_dir, config))
    p.start()
    p.join(timeout) 
    
    if p.is_alive():
        logger.error(f"Detection timed out after {timeout} seconds. Terminating process.")
        p.terminate()
        p.join()
        sys.exit(1) # Exit with error code

def _run_double_detection_worker(imageA_path, imageB_path, output_dir, config):
    logger.info(f"Running Cross-Image Detection on {imageA_path} and {imageB_path}")
    log_configuration(config)
    
    detector = CrossImageCopyDetector(config)
    try:
        imgA = imread2f(imageA_path)
        imgB = imread2f(imageB_path)
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        return

    try:
        logger.info("Starting cross-image detection pipeline...")
        start_time = time.time()
        maskA, maskB, clustersA, clustersB = detector.run(imgA, imgB, status_callback=status_logger)
        elapsed_time = time.time() - start_time
        logger.info(f"Detection completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return
    
    base_nameA = os.path.splitext(os.path.basename(imageA_path))[0]
    base_nameB = os.path.splitext(os.path.basename(imageB_path))[0]
    
    # Save Forgery Masks .npy
    # For double detection, we currently don't have the grouping logic implemented 
    # in the same way as single detection (which uses connected components of self-matches).
    # For now, we will save the raw clusters for A and B separately if possible, 
    # or just save an empty mask to avoid errors.
    
    # TODO: Implement grouping logic for cross-image detection
    logger.warning("Forgery mask grouping not yet implemented for double detection. Saving empty masks.")
    
    if detector.imageA is not None:
        shapeA = detector.imageA.shape[:2]
    else:
        shapeA = (0, 0) # Should not happen if run succeeded
        
    if detector.imageB is not None:
        shapeB = detector.imageB.shape[:2]
    else:
        shapeB = (0, 0)

    empty_maskA = np.zeros((0, shapeA[0], shapeA[1]), dtype=np.uint8)
    empty_maskB = np.zeros((0, shapeB[0], shapeB[1]), dtype=np.uint8)
    
    npy_pathA = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_forgery_maskA.npy")
    npy_pathB = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_forgery_maskB.npy")
    
    np.save(npy_pathA, empty_maskA)
    np.save(npy_pathB, empty_maskB)
    logger.info(f"Saved placeholder masks to {npy_pathA} and {npy_pathB}")

def run_double_detection(imageA_path, imageB_path, output_dir, config, timeout=600):
    p = multiprocessing.Process(target=_run_double_detection_worker, args=(imageA_path, imageB_path, output_dir, config))
    p.start()
    p.join(timeout) 
    
    if p.is_alive():
        logger.error(f"Detection timed out after {timeout} seconds. Terminating process.")
        p.terminate()
        p.join()
        sys.exit(1) # Exit with error code

def main():
    parser = argparse.ArgumentParser(description="Copy-Move Detection Tool - Forgery Mask Generator")
    parser.add_argument('--input', nargs='+', required=True, help="Input image path(s). Provide one for single detection, two for cross-image detection.")
    parser.add_argument('--output', required=True, help="Output directory for results.")
    parser.add_argument('--method', type=int, default=2, help="Feature extraction method ID (default: 2 for ZM-polar).")
    parser.add_argument('--timeout', type=int, default=600, help="Timeout in seconds for the detection process (default: 600).")
    
    # Clustering arguments
    parser.add_argument('--clustering_algo', type=str, default='dbscan', choices=['dbscan', 'hdbscan', 'optics', 'agglomerative', 'meanshift'], help="Clustering algorithm to use (default: dbscan).")
    parser.add_argument('--clustering_eps', type=float, default=0.5, help="Epsilon parameter for DBSCAN (default: 0.5).")
    parser.add_argument('--clustering_min_samples', type=int, default=13, help="Min samples parameter for clustering (default: 13).")
    parser.add_argument('--clustering_min_cluster_size', type=int, default=13, help="Min cluster size for HDBSCAN (default: 13).")
    parser.add_argument('--clustering_max_eps', type=float, default=float('inf'), help="Max epsilon for OPTICS (default: inf).")
    parser.add_argument('--clustering_xi', type=float, default=0.05, help="Xi parameter for OPTICS (default: 0.05).")
    parser.add_argument('--clustering_distance_threshold', type=float, default=0.5, help="Distance threshold for Agglomerative (default: 0.5).")
    parser.add_argument('--clustering_bandwidth_quantile', type=float, default=0.2, help="Quantile for MeanShift bandwidth estimation (default: 0.2).")

    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    config = {
        'type_feat': args.method,
        'clustering_algorithm': args.clustering_algo,
        'clustering_eps': args.clustering_eps,
        'clustering_min_samples': args.clustering_min_samples,
        'clustering_min_cluster_size': args.clustering_min_cluster_size,
        'clustering_max_eps': args.clustering_max_eps,
        'clustering_xi': args.clustering_xi,
        'clustering_distance_threshold': args.clustering_distance_threshold,
        'clustering_bandwidth_quantile': args.clustering_bandwidth_quantile,
    }
    
    if len(args.input) == 1:
        run_single_detection(args.input[0], args.output, config, args.timeout)
    elif len(args.input) == 2:
        run_double_detection(args.input[0], args.input[1], args.output, config, args.timeout)
    else:
        print("Error: Please provide either 1 or 2 input images.")
        sys.exit(1)

if __name__ == "__main__":
    main()
