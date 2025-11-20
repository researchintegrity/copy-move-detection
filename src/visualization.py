import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import cv2

class Visualizer:
    def __init__(self):
        pass

    def plot_matches(self, image, mask, mpfY, mpfX, padsize, clusters=None, max_lines=500, figsize=(12, 12), title="Matches"):
        """
        Visualize the matches by drawing lines between source and target regions.
        If clusters is provided, only matches within clusters are shown (outliers removed).
        """
        if image is None or mask is None or mpfY is None or mpfX is None:
            print("Missing data for visualization.")
            return

        # Filter mask if clusters are provided
        display_mask = mask
        if clusters is not None and len(clusters) > 0:
            combined_cluster_mask = np.zeros_like(mask, dtype=bool)
            for c in clusters:
                combined_cluster_mask |= c
            display_mask = combined_cluster_mask & mask

        plt.figure(figsize=figsize)
        plt.imshow(image)
        
        # Overlay the mask
        if padsize:
            mask_padded = np.pad(display_mask, padsize, 'constant', constant_values=False)
        else:
            mask_padded = display_mask
            
        plt.imshow(mask_padded, cmap='jet', alpha=0.3)

        # Draw lines
        y_start = padsize[0][0] if padsize else 0
        x_start = padsize[1][0] if padsize else 0

        y_indices, x_indices = np.where(display_mask)
        
        if len(y_indices) > 0:
            step = max(1, len(y_indices) // max_lines)
            
            for i in range(0, len(y_indices), step):
                y, x = y_indices[i], x_indices[i]
                
                # Target in feature map coords
                feat_dst_y = mpfY[y, x]
                feat_dst_x = mpfX[y, x]
                
                if np.isnan(feat_dst_y) or np.isnan(feat_dst_x):
                    continue

                # Source and Target in image coords
                src_y, src_x = y + y_start, x + x_start
                dst_y, dst_x = feat_dst_y + y_start, feat_dst_x + x_start
                
                plt.plot([src_x, dst_x], [src_y, dst_y], 'w-', linewidth=0.5, alpha=0.6)
                plt.plot(src_x, src_y, 'g.', markersize=3) # Source
                plt.plot(dst_x, dst_y, 'r.', markersize=3) # Target

        # plt.title(title)
        plt.axis('off')
        plt.show()

    def plot_clusters(self, image, clusters, padsize, mpfY=None, mpfX=None, figsize=(12, 12), title="Clusters"):
        """
        Visualize the clusters with different colors.
        If mpfY and mpfX are provided, clusters that are matching (connected) will share the same color.
        """
        if image is None or clusters is None:
            print("Missing data for visualization.")
            return

        # Ensure image is RGB for plotting
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)

        plt.figure(figsize=figsize)
        plt.imshow(image)
        
        if mpfY is None or mpfX is None:
            print("mpfY or mpfX not provided, skipping connected components analysis.")
            return

        # Create an empty RGBA image for the overlay
        overlay = np.zeros((image.shape[0], image.shape[1], 4))

        # 1. Compute Source and Target Masks for each cluster
        n_clusters = len(clusters)
        source_masks = []
        target_masks = []

        for i, src_mask in enumerate(clusters):
            source_masks.append(src_mask)
            
            # Compute Target Mask
            dst_mask = np.zeros_like(src_mask)
            
            # Vectorized version of user's loop
            y_idxs, x_idxs = np.where(src_mask)
            if len(y_idxs) > 0:
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
                     # post process dst_mask to remove outliers
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

        # 4. Visualize
        cmap = plt.get_cmap('tab10') 

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
            
            # Assign color
            color = cmap(group_id % 10)
            
            # Add to overlay
            overlay[group_mask_padded] = color
            overlay[group_mask_padded, 3] = 0.5

        plt.imshow(overlay)
        # plt.title(title)
        plt.axis('off')
        plt.show()
    
    def plot_matches_double(self, imageA, imageB, maskA, mpfYA, mpfXA, padsizeA, padsizeB, clustersA=None, max_lines=500, figsize=(18, 6), title="Cross-Image Matches"):
        """
        Visualize matches between Image A (left) and Image B (right).
        Only draws matches from A to B (using mpfYA, mpfXA).
        """
        if imageA is None or imageB is None or maskA is None:
            print("Missing data for visualization.")
            return

        # Ensure RGB
        if len(imageA.shape) == 2: imageA = np.stack((imageA,)*3, axis=-1)
        if len(imageB.shape) == 2: imageB = np.stack((imageB,)*3, axis=-1)

        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        
        # Create canvas
        canvas_h = max(hA, hB)
        canvas_w = wA + wB
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=imageA.dtype)
        canvas[:hA, :wA] = imageA
        canvas[:hB, wA:wA+wB] = imageB
        
        plt.figure(figsize=figsize)
        plt.imshow(canvas)
        plt.axvline(wA - 0.5, color='gray', linewidth=2)
        
        # Filter mask
        display_mask = maskA
        if clustersA is not None and len(clustersA) > 0:
            combined = np.zeros_like(maskA, dtype=bool)
            for c in clustersA:
                combined |= c
            display_mask = display_mask & combined
            
        # Draw lines
        y_start_A = padsizeA[0][0] if padsizeA else 0
        x_start_A = padsizeA[1][0] if padsizeA else 0
        y_start_B = padsizeB[0][0] if padsizeB else 0
        x_start_B = padsizeB[1][0] if padsizeB else 0
        
        y_indices, x_indices = np.where(display_mask)
        
        if len(y_indices) > 0:
            step = max(1, len(y_indices) // max_lines)
            
            for i in range(0, len(y_indices), step):
                y, x = y_indices[i], x_indices[i]
                
                # Target in feature map coords of B
                feat_dst_y = mpfYA[y, x]
                feat_dst_x = mpfXA[y, x]
                
                if np.isnan(feat_dst_y) or np.isnan(feat_dst_x):
                    continue
                    
                # Source in Image A
                src_y = y + y_start_A
                src_x = x + x_start_A
                
                # Target in Image B
                dst_y = feat_dst_y + y_start_B
                dst_x = feat_dst_x + x_start_B
                
                # Adjust target x for canvas (shift by wA)
                dst_x_canvas = dst_x + wA
                
                plt.plot([src_x, dst_x_canvas], [src_y, dst_y], 'w-', linewidth=0.5, alpha=0.6)
                plt.plot(src_x, src_y, 'g.', markersize=3)
                plt.plot(dst_x_canvas, dst_y, 'r.', markersize=3)
                
        plt.axis('off')
        plt.show()
    
    def plot_clusters_double(self, imageA, imageB, clustersA, clustersB, mpfYA, mpfXA, mpfYB, mpfXB, padsizeA, padsizeB, figsize=(18, 6), title="Cross-Image Clusters"):
        """
        Visualize clusters on both images, coloring connected components (A <-> B) with the same color.
        """
        if imageA is None or imageB is None:
            print("Missing images for visualization.")
            return
            
        # Ensure RGB
        if len(imageA.shape) == 2: imageA = np.stack((imageA,)*3, axis=-1)
        if len(imageB.shape) == 2: imageB = np.stack((imageB,)*3, axis=-1)
        
        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        
        # Create canvas
        canvas_h = max(hA, hB)
        canvas_w = wA + wB
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=imageA.dtype)
        canvas[:hA, :wA] = imageA
        canvas[:hB, wA:wA+wB] = imageB
        
        # Prepare overlays
        overlay = np.zeros((canvas_h, canvas_w, 4))
        
        # Handle cases where clusters might be None
        clustersA = clustersA if clustersA is not None else []
        clustersB = clustersB if clustersB is not None else []
        
        nA = len(clustersA)
        nB = len(clustersB)
        n_total = nA + nB
        
        if n_total == 0:
            plt.figure(figsize=figsize)
            plt.imshow(canvas)
            plt.axis('off')
            plt.show()
            return

        # Helper to compute target mask
        def get_target_mask(src_mask, mpfY, mpfX, shape_dst):
            dst_mask = np.zeros(shape_dst, dtype=bool)
            y_idxs, x_idxs = np.where(src_mask)
            if len(y_idxs) > 0:
                val_ys = mpfY[y_idxs, x_idxs]
                val_xs = mpfX[y_idxs, x_idxs]
                
                valid = ~np.isnan(val_ys) & ~np.isnan(val_xs)
                tys = np.round(val_ys[valid]).astype(int)
                txs = np.round(val_xs[valid]).astype(int)
                
                H, W = shape_dst
                in_bounds = (tys >= 0) & (tys < H) & (txs >= 0) & (txs < W)
                tys = tys[in_bounds]
                txs = txs[in_bounds]
                
                dst_mask[tys, txs] = True
                # Morphological closing to fill gaps
                dst_mask = cv2.morphologyEx(dst_mask.astype(np.uint8), cv2.MORPH_CLOSE, 
                                            np.ones((5, 5), np.uint8)).astype(bool)
            return dst_mask

        # 1. Compute Mappings
        # A -> B
        targets_A_in_B = []
        shape_feat_B = mpfYB.shape # Assuming feature map size matches mpf
        for mask in clustersA:
            targets_A_in_B.append(get_target_mask(mask, mpfYA, mpfXA, shape_feat_B))
            
        # B -> A
        targets_B_in_A = []
        shape_feat_A = mpfYA.shape
        for mask in clustersB:
            targets_B_in_A.append(get_target_mask(mask, mpfYB, mpfXB, shape_feat_A))
            
        # 2. Build Adjacency Matrix
        adj = np.zeros((n_total, n_total), dtype=int)
        
        # Check overlaps between A clusters and B clusters
        # Nodes 0..nA-1 are A clusters
        # Nodes nA..nA+nB-1 are B clusters
        
        for i in range(nA):
            target_mask_B = targets_A_in_B[i]
            for j in range(nB):
                cluster_mask_B = clustersB[j]
                # Check overlap in B domain
                overlap = np.logical_and(target_mask_B, cluster_mask_B).sum()
                if overlap > 5:
                    adj[i, nA + j] = 1
                    adj[nA + j, i] = 1
                    
        # Also check B->A mappings for robustness
        for j in range(nB):
            target_mask_A = targets_B_in_A[j]
            for i in range(nA):
                cluster_mask_A = clustersA[i]
                # Check overlap in A domain
                overlap = np.logical_and(target_mask_A, cluster_mask_A).sum()
                if overlap > 5:
                    adj[nA + j, i] = 1
                    adj[i, nA + j] = 1
                    
        # 3. Connected Components
        n_components, labels = connected_components(csr_matrix(adj), directed=False)
        
        # 4. Visualize
        cmap = plt.get_cmap('tab10')
        
        for group_id in range(n_components):
            color = cmap(group_id % 10)
            
            # Get nodes in this component
            nodes = np.where(labels == group_id)[0]
            
            # Process A nodes
            nodes_A = nodes[nodes < nA]
            if len(nodes_A) > 0:
                group_mask_A = np.zeros(mpfYA.shape, dtype=bool)
                for idx in nodes_A:
                    group_mask_A |= clustersA[idx]
                
                if padsizeA:
                    mask_padded_A = np.pad(group_mask_A, padsizeA, 'constant', constant_values=False)
                else:
                    mask_padded_A = group_mask_A
                    
                # Add to overlay (left side)
                overlay[:hA, :wA][mask_padded_A] = color
                overlay[:hA, :wA][mask_padded_A, 3] = 0.5
                
            # Process B nodes
            nodes_B = nodes[nodes >= nA] - nA
            if len(nodes_B) > 0:
                group_mask_B = np.zeros(mpfYB.shape, dtype=bool)
                for idx in nodes_B:
                    group_mask_B |= clustersB[idx]
                    
                if padsizeB:
                    mask_padded_B = np.pad(group_mask_B, padsizeB, 'constant', constant_values=False)
                else:
                    mask_padded_B = group_mask_B
                    
                # Add to overlay (right side, shifted by wA)
                # Need to be careful with indices
                overlay[:hB, wA:wA+wB][mask_padded_B] = color
                overlay[:hB, wA:wA+wB][mask_padded_B, 3] = 0.5
                
        plt.figure(figsize=figsize)
        plt.imshow(canvas)
        plt.imshow(overlay)
        plt.axvline(wA - 0.5, color='gray', linewidth=2)
        # plt.title(title)
        plt.axis('off')
        plt.show()
