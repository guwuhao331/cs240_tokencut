import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from PIL import Image

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except ImportError:
    print("Warning: pydensecrf not found. CRF post-processing will not work.")
    dcrf = None


def density_aware_affinity(features, k=10, t0=1.0, alpha=0.5, tau=0.15, eps=1e-5):
    """
    [Paper Section 3.2] Density-Tune Cosine Similarity (Eq. 3 & 4)
    Computes affinity matrix with adaptive temperature based on local feature density.
    """
    N = features.shape[1]
    
    # 1. Compute standard cosine similarity matrix S
    # features: (D, N), normalized
    S = features.T @ features # (N, N)
    
    # 2. Compute local density rho_i (Eq. 4)
    # Select top-k neighbors (excluding self)
    # We use top-(k+1) and exclude the first one (self)
    topk_vals, _ = torch.topk(S, k=k+1, dim=1) 
    # Sum of similarities of k neighbors / k
    rho = torch.mean(topk_vals[:, 1:], dim=1) # (N,)
    
    # 3. Compute adaptive temperature T_ij
    # T_ij = T0 + alpha * (rho_i + rho_j) / 2
    rho_matrix = (rho.unsqueeze(0) + rho.unsqueeze(1)) / 2 # (N, N)
    T = t0 + alpha * rho_matrix
    
    # 4. Compute modulated affinity W_ij (Eq. 3)
    W = S / T
    
    # 5. Thresholding (consistent with TokenCut logic mentioned in paper)
    A = (W > tau).float()
    A[A == 0] = eps
    
    return A


def boundary_augmentation(eigen_vec, h, w, k=8):
    """
    [Paper Section 3.2] Boundary Augmentation (Eq. 5 & 6)
    Enhances the eigenvector by subtracting local boundary information.
    """
    # X: (N,) -> (H, W)
    X = eigen_vec.view(h, w)
    
    # Pad to handle boundaries (replicate padding as per paper)
    pad_size = 1 # k=8 implies 3x3 neighborhood, radius 1
    X_pad = F.pad(X.unsqueeze(0).unsqueeze(0), (pad_size, pad_size, pad_size, pad_size), mode='replicate').squeeze()
    
    # Calculate X_b: average absolute difference with neighbors (Eq. 6)
    # We implement this efficiently using unfolded patches or manual shifting
    diff_sum = torch.zeros_like(X)
    count = 0
    
    # Iterate over 8 neighbors
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            
            # Extract neighbor window
            neighbor = X_pad[1+dy : 1+dy+h, 1+dx : 1+dx+w]
            diff_sum += torch.abs(X - neighbor)
            count += 1
            
    X_b = diff_sum / count
    
    # Calculate X_a = X - X_b (Eq. 5)
    X_a = X - X_b
    
    return X_a.flatten()


def ncut_coler(features, dims, tau=0.15, eps=1e-5):
    """
    Full CutOnce Ncut implementation with Density-Tune and Boundary Augmentation.
    """
    h, w = dims
    feats = F.normalize(features, p=2, dim=0)
    N = feats.shape[1]

    # --- Module 1: Density-Tune Cosine Similarity ---
    # Default parameters from paper: k=10, T0=1.0, alpha=0.5
    A = density_aware_affinity(feats, k=10, t0=1.0, alpha=0.5, tau=tau, eps=eps)

    # Standard Ncut: Solve (D - W)x = lambda D x
    d = A.sum(dim=1)
    inv_sqrt_d = d.pow(-0.5)
    D_inv_sqrt = torch.diag(inv_sqrt_d)
    
    L_sym = torch.eye(N, device=features.device) - D_inv_sqrt @ A @ D_inv_sqrt

    # Eigen decomposition
    # Note: eigh is used as L_sym is symmetric
    eigenvalues, eigenvectors = torch.linalg.eigh(L_sym, UPLO='L')
    second_vec = eigenvectors[:, 1] # Fiedler vector
    
    # --- Module 2: Boundary Augmentation ---
    # Enhance the second eigenvector
    enhanced_vec = boundary_augmentation(second_vec, h, w)
    
    return enhanced_vec


def get_masks_coler(eigen_vec, dims, tau_filter=0.95, object_centric=True):
    """
    [Paper Section 3.2] Ranking-Based Instance Filter
    """
    h, w = dims
    masks = []
    
    # Thresholding
    avg = torch.mean(eigen_vec)
    bipartition = eigen_vec > avg
    
    # Corner check (Object Centric Prior)
    bipartition_reshaped = bipartition.reshape(h, w).cpu().numpy()
    corners = int(bipartition_reshaped[0,0]) + int(bipartition_reshaped[0,-1]) + \
              int(bipartition_reshaped[-1,0]) + int(bipartition_reshaped[-1,-1])
    
    if corners >= 3 and object_centric:
        eigen_vec = -eigen_vec
        bipartition = np.logical_not(bipartition_reshaped)
    else:
        bipartition = bipartition_reshaped

    bipartition = ndimage.binary_fill_holes(bipartition)
    objects, num_objects = ndimage.label(bipartition)
    
    if num_objects < 1:
        return []

    labels, counts = np.unique(objects, return_counts=True)
    
    # Calculate feature sums for each component (S_i in paper)
    # Note: Using enhanced eigenvector values for summing
    eigen_vec_np = eigen_vec.cpu().numpy().reshape(h, w)
    sums = ndimage.sum(eigen_vec_np, labels=objects, index=labels)

    # Filter background (label 0)
    if labels[0] == 0:
        valid_labels = labels[1:]
        valid_sums = sums[1:]
    else:
        valid_labels = labels
        valid_sums = sums

    if len(valid_sums) == 0:
        return []

    # --- Module 3: Ranking-Based Filter ---
    # 1. Sort by feature sum descending
    order = np.argsort(valid_sums)[::-1]
    sorted_sums = valid_sums[order]
    sorted_labels = valid_labels[order]
    
    # 2. Cumulative screening
    # Select objects until cumulative sum >= tau * total_sum
    total_sum = np.sum(sorted_sums)
    current_sum = 0
    
    for idx, s in enumerate(sorted_sums):
        current_sum += s
        label_id = sorted_labels[idx]
        mask = (objects == label_id).astype(np.uint8)
        masks.append(mask)
        
        # Check cumulative threshold (tau=0.95 default)
        if (current_sum / total_sum) >= tau_filter:
            break
            
    return masks


def bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax - cmin + 1, rmax - rmin + 1


def densecrf_post_process(image_np, mask_np):
    """
    CRF Post-processing (Standard).
    Used in 'CutOnce+' or 'COLER' training generation, but technically optional for pure CutOnce.
    """
    if dcrf is None:
        return mask_np

    mask_np = mask_np.astype(np.float32)
    h_orig, w_orig = mask_np.shape
    
    bbox = bbox_from_mask(mask_np)
    if bbox is None:
        return mask_np
        
    x, y, w, h = bbox
    pad_x = int(w * 0.33)
    pad_y = int(h * 0.33)
    
    x1 = max(x - pad_x, 0)
    x2 = min(x + w + pad_x, w_orig)
    y1 = max(y - pad_y, 0)
    y2 = min(y + h + pad_y, h_orig)
    
    mask_crop = mask_np[y1:y2, x1:x2]
    image_crop = image_np[y1:y2, x1:x2, :] 
    
    h_crop, w_crop = mask_crop.shape
    
    prob = np.zeros((2, h_crop, w_crop), dtype=np.float32)
    prob[0, :, :] = np.where(mask_crop == 0, 0.9, 0.1)
    prob[1, :, :] = np.where(mask_crop == 1, 0.9, 0.1)
    
    d = dcrf.DenseCRF2D(w_crop, h_crop, 2)
    U = unary_from_softmax(prob)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=7)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(image_crop), compat=10)
    
    Q = d.inference(10)
    map_crop = np.argmax(Q, axis=0).reshape((h_crop, w_crop))
    map_crop = ndimage.binary_fill_holes(map_crop).astype(np.uint8)
    
    mask_final = np.zeros_like(mask_np)
    mask_final[y1:y2, x1:x2] = map_crop
    
    intersection = np.sum((mask_np * mask_final) > 0.5)
    union = np.sum((mask_np + mask_final) > 0.5)
    iou = intersection / (union + 1e-6)
    
    if np.sum(mask_final) == 0 or iou < 0.5:
        return mask_np
        
    return mask_final