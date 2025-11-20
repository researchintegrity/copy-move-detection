import numpy as np

def dat2info(c):
    """
    Convert affine parameters to geometric info (scale, flip, rotation, etc.)
    c: [bx, ax, cx, px, by, ay, cy, py]
    """
    # A is the linear part of the affine transformation
    # [ ax  bx ]
    # [ ay  by ]
    # Note: The order in c is [bx, ax, cx, px, by, ay, cy, py]
    # So c[0]=bx, c[1]=ax, c[4]=by, c[5]=ay
    # Wait, let's check core.py usage:
    # A = np.asarray([[c[0], c[1]], [c[4], c[5]]])
    # If c corresponds to [bx, ax, cx, px, by, ay, cy, py]
    # Then A = [[bx, ax], [by, ay]]
    
    A = np.asarray([[c[0], c[1]], [c[4], c[5]]])
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
    s = np.mean(s)
    u = u @ vh
    f = np.sign(np.linalg.det(u))
    u = u @ np.asarray([[f, 0], [0, 1]])
    phi = np.arctan2(u[1, 0] - u[0, 1], u[0, 0] + u[1, 1]) * 180 / np.pi
    
    cx = c[3] # px
    cy = c[7] # py
    dx = c[2] # cx (offset x)
    dy = c[6] # cy (offset y)
    
    Ap = s * u @ np.asarray([[f, 0], [0, 1]])
    e = np.mean((A - Ap)**2)
    
    return [s, f, phi, cx, cy, dx, dy, e] # scale, flip, angle, cx, cy, dx, dy, mse

def get_info_text(mask, c):
    [s, f, phi, cx, cy, dx, dy, e] = dat2info(c)
    d = np.sum(mask)
    if f < 0:
        if np.abs(phi) < 90:
            text = ' Scale\t: %.2f\n Flip\t: %s\n Rotate\t: %.0f\n Size\t: %d' % (s, 'Horizontal', phi, d)
        elif phi < 0:
            text = ' Scale\t: %.2f\n Flip\t: %s\n Rotate\t: %.0f\n Size\t: %d' % (s, 'Vertical', phi + 180, d)
        else:
            text = ' Scale\t: %.2f\n Flip\t: %s\n Rotate\t: %.0f\n Size\t: %d' % (s, 'Vertical', phi - 180, d)
    else:
        text = ' Scale\t: %.2f\n Flip\t: %s\n Rotate\t: %.0f\n Size\t: %d' % (s, 'None', phi, d)
                  
    c_centro = np.asarray((cx, cy))
    c_end = np.asarray((dx, dy))
    c_delta = c_end - c_centro
    c_delta0 = c_delta / (np.max(np.abs(c_delta)) + 1e-9)
    
    c_delta1 = np.asarray((c_delta0[1], -c_delta0[0]))
    c_delta = c_delta1 @ np.asarray([[1, -1], [1, 1]])
    c_delta = c_delta / (np.max(np.abs(c_delta)) + 1e-9)
    quadrante = int((np.sign(c_delta[1]) + 1) // 2 + (np.sign(c_delta[0]) + 1))
    
    return text, c_centro, quadrante

class Analyzer:
    def __init__(self):
        pass

    def analyze_clusters(self, clusters, cluster_centers):
        """
        Analyze each cluster to extract geometric transformation info.
        """
        results = []
        for i, mask in enumerate(clusters):
            if i >= len(cluster_centers):
                break
            
            center_data = cluster_centers[i]
            text, center_point, quadrant = get_info_text(mask, center_data)
            
            results.append({
                'cluster_id': i,
                'text': text,
                'center_point': center_point,
                'quadrant': quadrant,
                'raw_data': center_data
            })
        return results
