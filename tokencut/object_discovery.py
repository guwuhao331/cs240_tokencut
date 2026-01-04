"""
Main functions for applying Normalized Cut.
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import torch
import torch.nn.functional as F
import numpy as np
# from scipy.linalg import eigh
from scipy import ndimage
from .fastncut import FastNcut


def _eigh_symmetric(matrix: torch.Tensor, uplo: str = "U"):
    # Prefer GPU for speed; fall back to CPU if MAGMA/GPU eigensolver fails.
    linalg = getattr(torch, "linalg", None)
    if linalg is not None and hasattr(linalg, "eigh"):
        try:
            return linalg.eigh(matrix, UPLO=uplo)
        except Exception:
            pass
    if hasattr(torch, "symeig"):
        try:
            return torch.symeig(matrix, eigenvectors=True)
        except Exception:
            pass

    matrix_np = matrix.detach().cpu().numpy()
    values, vectors = np.linalg.eigh(matrix_np)
    values_t = torch.from_numpy(values)
    vectors_t = torch.from_numpy(vectors)
    if matrix.is_cuda:
        values_t = values_t.to(matrix.device)
        vectors_t = vectors_t.to(matrix.device)
    return values_t, vectors_t


def ncut(
    feats,
    dims,
    scales,
    init_image_size,
    tau=0,
    eps=1e-5,
    im_name="",
    no_binary_graph=False,
):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    cls_token = feats[0, 0:1, :].detach()

    feats = feats[0, 1:, :]
    feats = torch.nn.functional.normalize(feats, p=2)
    A = feats @ feats.transpose(1, 0)

    if no_binary_graph:
        A[A < tau] = eps
    else:
        A = (A > tau).float()
        A[A == 0] = eps
    d_i = torch.sum(A, dim=1)
    D = torch.diag(d_i)

    # Print second and third smallest eigenvector
    L = D - A
    _, eigenvectors = _eigh_symmetric(L, uplo="U")
    eigenvectors = eigenvectors.cpu().numpy()
    eigenvec = np.copy(eigenvectors[:, 0])

    # Using average point to compute bipartition
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg

    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _, objects, cc = detect_box(
        bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]
    )  ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0], cc[1]] = 1

    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)


# def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
#     """
#     Implementation of NCut Method.
#     Inputs
#       feats: the pixel/patche features of an image
#       dims: dimension of the map from which the features are used
#       scales: from image to map scale
#       init_image_size: size of the image
#       tau: thresold for graph construction
#       eps: graph edge weight
#       im_name: image_name
#       no_binary_graph: ablation study for using similarity score as graph edge weight
#     """
#     cls_token = feats[0,0:1,:].cpu().numpy() 

#     feats = feats[0,1:,:]
#     feats = F.normalize(feats, p=2)
#     A = (feats @ feats.transpose(1,0)) 
#     A = A.cpu().numpy()
#     if no_binary_graph:
#         A[A<tau] = eps
#     else:
#         A = A > tau
#         A = np.where(A.astype(float) == 0, eps, A)
#     d_i = np.sum(A, axis=1)
#     D = np.diag(d_i)
  
#     # Print second and third smallest eigenvector 
#     _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
#     eigenvec = np.copy(eigenvectors[:, 0])

#     # Using average point to compute bipartition 
#     second_smallest_vec = eigenvectors[:, 0]
#     avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
#     bipartition = second_smallest_vec > avg
    
#     seed = np.argmax(np.abs(second_smallest_vec))

#     if bipartition[seed] != 1:
#         eigenvec = eigenvec * -1
#         bipartition = np.logical_not(bipartition)
#     bipartition = bipartition.reshape(dims).astype(float)

#     # predict BBox
#     pred, _, objects,cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]) ## We only extract the principal object BBox
#     mask = np.zeros(dims)
#     mask[cc[0],cc[1]] = 1

#     return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)

def detect_box(
    bipartition, seed, dims, initial_im_size=None, scales=None, principle_object=True
):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]

    if principle_object:
        mask = np.where(objects == cc)
        # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError


def fast_ncut(
    feats,
    dims,
    scales,
    init_image_size,
    tau=0,
    eps=1e-5,
    im_name="",
    no_binary_graph=False,
):
    """
    Implementation of Fast NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    cls_token = feats[0, 0:1, :].cpu().numpy()

    feats = feats[0, 1:, :]
    feats = F.normalize(feats, p=2)
    A = feats @ feats.transpose(1, 0)
    A = A.cpu().numpy()
    if no_binary_graph:
        A[A < tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)

    const = np.array([[0, 1], [0, 2], [1, 5], [3, 8], [24, 33]])
    model = FastNcut(const=const, A=A)
    fast_ncut_result = model.fit(A)
    second_smallest_vec = fast_ncut_result.reshape(-1)
    eigenvec = fast_ncut_result.reshape(-1)

    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg

    seed = np.argmin(np.abs(second_smallest_vec))

    # if bipartition[seed] != 1:
    #     eigenvec = eigenvec * -1
    #     bipartition = np.logical_not(bipartition)
    eigenvec = eigenvec * -1
    bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _, objects, cc = detect_box(
        bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]
    )  ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0], cc[1]] = 1

    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)


def fast_ncut_optimized(
    feats,
    dims,
    scales,
    init_image_size,
    tau=0,
    eps=1e-5,
    im_name="",
    no_binary_graph=False,
):
    """
    Further optimized implementation of Fast NCut Method.
    """

    # torch acc
    feats = feats[0]
    cls_token = feats[0:1, :].detach()

    # standardize
    feats = torch.nn.functional.normalize(feats[1:, :], p=2)

    # similarity matrix, torch tensor calculation
    A = feats @ feats.t()

    # optional binary graph
    if no_binary_graph:
        A = torch.where(A < tau, eps, A)
    else:
        A = (A > tau).float()
        A[A == 0] = eps

    A_numpy = A.cpu().numpy()

    const = np.array([[0, 1], [0, 2], [1, 5], [3, 8], [4, 9]])
    model = FastNcut(const=const, A=A_numpy)
    fast_ncut_result = model.fit(A_numpy)
    second_smallest_vec = fast_ncut_result.reshape(-1)

    # mean and bipartition
    avg = second_smallest_vec.mean()
    bipartition = second_smallest_vec > avg
    eigenvec = -fast_ncut_result.reshape(-1)
    bipartition = np.logical_not(bipartition).astype(float).reshape(dims)

    # optimal
    seed = np.argmin(np.abs(second_smallest_vec))
    pred, _, objects, cc = detect_box(
        bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]
    )
    mask = np.zeros(dims)
    mask[cc[0], cc[1]] = 1

    return np.asarray(pred), objects, mask, None, eigenvec.reshape(dims)
