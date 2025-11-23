import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import multiprocessing
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_detection")

try:
    from copy_move_detection.detector import CopyMoveDetector, CrossImageCopyDetector
    from copy_move_detection.utility.utilityImage import imread2f
except ImportError:
    # Ensure we can import from src locally
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.detector import CopyMoveDetector, CrossImageCopyDetector
    from src.utility.utilityImage import imread2f

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

def save_visualization(fig, output_path):
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

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

    # Save Mask
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    logger.info(f"Saved mask to {mask_path}")
    
    # Visualize and Save Matches
    detector.visualize_matches()
    matches_path = os.path.join(output_dir, f"{base_name}_matches.png")
    save_visualization(plt.gcf(), matches_path)
    logger.info(f"Saved matches visualization to {matches_path}")
    
    # Visualize and Save Clusters
    detector.visualize_clusters()
    clusters_path = os.path.join(output_dir, f"{base_name}_clusters.png")
    save_visualization(plt.gcf(), clusters_path)
    logger.info(f"Saved clusters visualization to {clusters_path}")

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
    
    # Save Masks
    maskA_path = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_maskA.png")
    maskB_path = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_maskB.png")
    cv2.imwrite(maskA_path, (maskA * 255).astype(np.uint8))
    cv2.imwrite(maskB_path, (maskB * 255).astype(np.uint8))
    logger.info(f"Saved masks to {maskA_path} and {maskB_path}")
    
    # Visualize and Save Matches
    detector.visualize_matches()
    matches_path = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_matches.png")
    save_visualization(plt.gcf(), matches_path)
    logger.info(f"Saved matches visualization to {matches_path}")
    
    # Visualize and Save Clusters
    detector.visualize_clusters()
    clusters_path = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_clusters.png")
    save_visualization(plt.gcf(), clusters_path)
    logger.info(f"Saved clusters visualization to {clusters_path}")

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
    parser = argparse.ArgumentParser(description="Copy-Move Detection Tool")
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
