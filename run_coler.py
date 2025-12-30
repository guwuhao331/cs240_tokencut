import os
import argparse
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from networks import get_coler_model 
# Import the strictly implemented COLER functions
from cutonce import ncut_coler, get_masks_coler, densecrf_post_process

def resize_pil(I, patch_size=16, resize_long_edge=None):
    if resize_long_edge is not None and resize_long_edge > 0:
        w, h = I.size
        if max(w, h) > resize_long_edge:
            scale = resize_long_edge / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            I = I.resize((new_w, new_h), resample=Image.LANCZOS)
    
    w, h = I.size
    new_w = (w // patch_size) * patch_size
    new_h = (h // patch_size) * patch_size
    
    if new_w != w or new_h != h:
        I = I.resize((new_w, new_h), resample=Image.LANCZOS)
        
    return I, new_w, new_h

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.img_path):
        print(f"Error: Image not found at {args.img_path}")
        return

    # 1. Load Image
    img_orig = Image.open(args.img_path).convert('RGB')
    w_orig, h_orig = img_orig.size
    img_orig_np = np.array(img_orig)
    print(f"Original size: {w_orig}x{h_orig}")

    # 2. Resize for Model
    img_model = img_orig.copy()
    img_model, w_model, h_model = resize_pil(img_model, args.patch_size, resize_long_edge=args.resize_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    input_tensor = transform(img_model).unsqueeze(0).to(device)

    # 3. Extract Features
    print(f"Loading model {args.arch}...")
    model = get_coler_model(arch=args.arch, patch_size=args.patch_size, device=device)
    model.eval()

    print("Extracting features...")
    start_time = time.time()
    with torch.no_grad():
        q, k, v = model.get_last_features(input_tensor)
        feat = k[0] 
        feat = feat[:, 1:, :] 
        feat = feat.permute(0, 2, 1)
        feat = feat.reshape(-1, feat.shape[-1]) 

    feat_w = w_model // args.patch_size
    feat_h = h_model // args.patch_size
    
    # 4. Run COLER CutOnce
    # Using the strict implementation from the paper
    print("Running CutOnce (Density-Tune + Boundary Augmentation)...")
    # tau=0.15 is default from paper for Eq. 3
    eigen_vec = ncut_coler(feat, (feat_h, feat_w), tau=args.tau)

    # 5. Generate Masks
    print("Filtering Instances (Ranking-Based)...")
    # tau_filter=0.95 is default from paper for ranking
    masks_low_res = get_masks_coler(eigen_vec, (feat_h, feat_w), tau_filter=0.95)
    print(f"Found {len(masks_low_res)} objects.")

    # 6. Visualization & CRF
    plt.figure(figsize=(10, 5))
    plt.subplot(1, len(masks_low_res) + 1, 1)
    plt.imshow(img_orig)
    plt.title("Original")
    plt.axis('off')

    for i, mask in enumerate(masks_low_res):
        # Upscale
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_up = torch.nn.functional.interpolate(
            mask_tensor, size=(h_orig, w_orig), mode='nearest'
        ).squeeze().numpy()

        # Optional CRF (CutOnce+ mode)
        if args.use_crf:
            try:
                mask_final = densecrf_post_process(img_orig_np, mask_up)
            except Exception as e:
                print(f"CRF failed: {e}")
                mask_final = mask_up
        else:
            mask_final = mask_up

        plt.subplot(1, len(masks_low_res) + 1, i + 2)
        plt.imshow(mask_final)
        plt.title(f"Obj {i+1}")
        plt.axis('off')

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.4f}s")

    out_path = "coler_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Result saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--arch", type=str, default="vit_base")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.15) # Default 0.15 per paper
    parser.add_argument("--use_crf", action='store_true', help="Enable CRF (CutOnce+)")
    parser.add_argument("--resize_size", type=int, default=480, help="Resize long edge")
    
    args = parser.parse_args()
    main(args)