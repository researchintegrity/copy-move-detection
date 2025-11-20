import numpy as np

def get_default_parameters():
    return {
        # Feature Extraction
        'type_feat': 2, # 1: ZM-cart, 2: ZM-polar, 3: PCT-cart, 4: PCT-polar, 5: FMT
        'ZM_order': 5,
        'radiusNum': 26,
        'anglesNum': 32,
        'radiusMin': np.sqrt(2.0),
        'pad_img': False,
        
        # Matching
        'match_num_iter': 8,
        'match_th_dist1': 8,
        'match_num_tile': 1,
        'match_diameter': 1, # block-size for matching
        
        # Post-processing / Filtering
        'th2_dist2': 10 * 10,
        'th2_dlf': 300,
        'th_scale': 0.05,
        'th_sizeA': 300,
        'th_sizeB': 300,
        'rd_median': 4,
        'rd_dlf': 6,
        'rd_dil': 10, # usually rd_dlf + rd_median
        'rd_fit': 3,
        
        # Clustering
        'fact_pos': 100,
        'clustering_rd_dil': 10,
    }
