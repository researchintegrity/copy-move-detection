import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from copy_move_detection.detector import CopyMoveDetector, CrossImageCopyDetector
    from copy_move_detection.utility.utilityImage import imread2f
except ImportError:
    # Ensure we can import from src locally
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.detector import CopyMoveDetector, CrossImageCopyDetector
    from src.utility.utilityImage import imread2f

def save_visualization(fig, output_path):
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def run_single_detection(image_path, output_dir, config):
    print(f"Running Single Image Detection on {image_path}")
    
    detector = CopyMoveDetector(config)
    try:
        img = imread2f(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # Run detection
    mask, clusters = detector.run(img)
    
    # Save Mask
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    print(f"Saved mask to {mask_path}")
    
    # Visualize and Save Matches
    detector.visualize_matches()
    matches_path = os.path.join(output_dir, f"{base_name}_matches.png")
    save_visualization(plt.gcf(), matches_path)
    print(f"Saved matches visualization to {matches_path}")
    
    # Visualize and Save Clusters
    detector.visualize_clusters()
    clusters_path = os.path.join(output_dir, f"{base_name}_clusters.png")
    save_visualization(plt.gcf(), clusters_path)
    print(f"Saved clusters visualization to {clusters_path}")

def run_double_detection(imageA_path, imageB_path, output_dir, config):
    print(f"Running Cross-Image Detection on {imageA_path} and {imageB_path}")
    
    detector = CrossImageCopyDetector(config)
    try:
        imgA = imread2f(imageA_path)
        imgB = imread2f(imageB_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Run detection
    maskA, maskB, clustersA, clustersB = detector.run(imgA, imgB)
    
    base_nameA = os.path.splitext(os.path.basename(imageA_path))[0]
    base_nameB = os.path.splitext(os.path.basename(imageB_path))[0]
    
    # Save Masks
    maskA_path = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_maskA.png")
    maskB_path = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_maskB.png")
    cv2.imwrite(maskA_path, (maskA * 255).astype(np.uint8))
    cv2.imwrite(maskB_path, (maskB * 255).astype(np.uint8))
    print(f"Saved masks to {maskA_path} and {maskB_path}")
    
    # Visualize and Save Matches
    detector.visualize_matches()
    matches_path = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_matches.png")
    save_visualization(plt.gcf(), matches_path)
    print(f"Saved matches visualization to {matches_path}")
    
    # Visualize and Save Clusters
    detector.visualize_clusters()
    clusters_path = os.path.join(output_dir, f"{base_nameA}_vs_{base_nameB}_clusters.png")
    save_visualization(plt.gcf(), clusters_path)
    print(f"Saved clusters visualization to {clusters_path}")

def main():
    parser = argparse.ArgumentParser(description="Copy-Move Detection Tool")
    parser.add_argument('--input', nargs='+', required=True, help="Input image path(s). Provide one for single detection, two for cross-image detection.")
    parser.add_argument('--output', required=True, help="Output directory for results.")
    parser.add_argument('--method', type=int, default=2, help="Feature extraction method ID (default: 2 for ZM-polar).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    config = {
        'type_feat': args.method,
    }
    
    if len(args.input) == 1:
        run_single_detection(args.input[0], args.output, config)
    elif len(args.input) == 2:
        run_double_detection(args.input[0], args.input[1], args.output, config)
    else:
        print("Error: Please provide either 1 or 2 input images.")
        sys.exit(1)

if __name__ == "__main__":
    main()
