import os
import argparse
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from networks import get_coler_model 
from cutonce import ncut_coler, get_masks_coler, densecrf_post_process

def pad_to_patch_size(I, patch_size=8):
    """
    Pads the image so that dimensions are divisible by patch_size.
    Does NOT resize or downsample the image, maintaining full resolution.
    """
    w, h = I.size
    new_w = int(np.ceil(w / patch_size) * patch_size)
    new_h = int(np.ceil(h / patch_size) * patch_size)
    
    if new_w != w or new_h != h:
        # Create a new padded image (black padding)
        I_new = Image.new('RGB', (new_w, new_h), (0, 0, 0))
        I_new.paste(I, (0, 0))
        return I_new, new_w, new_h, w, h
    
    return I, w, h, w, h

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.img_path):
        print(f"Error: Image not found at {args.img_path}")
        return

    # 1. Load Original Image
    img_orig = Image.open(args.img_path).convert('RGB')
    w_orig, h_orig = img_orig.size
    print(f"Original image size: {w_orig}x{h_orig}")

    # 2. Prepare Model Input (FULL RESOLUTION)
    # Replaced resizing with padding to prevent pooling/loss of detail.
    img_model, w_model, h_model, valid_w, valid_h = pad_to_patch_size(img_orig, args.patch_size)
    print(f"Model input size: {w_model}x{h_model} (No Downsampling)")

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
    
    try:
        with torch.no_grad():
            q, k, v = model.get_last_features(input_tensor)
            feat = k[0] 
            feat = feat[:, 1:, :] 
            feat = feat.permute(0, 2, 1)
            feat = feat.reshape(-1, feat.shape[-1]) 
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nError: CUDA Out of Memory!")
            print("The image is too large for full-resolution processing on this GPU.")
            return
        else:
            raise e

    feat_w = w_model // args.patch_size
    feat_h = h_model // args.patch_size
    print(f"Feature map size: {feat_h}x{feat_w}")

    # 4. Run CutOnce (COLER)
    print("Running CutOnce (Density-Tune + Boundary Augmentation)...")
    eigen_vec = ncut_coler(feat, (feat_h, feat_w), tau=args.tau)

    # 5. Generate Masks
    print("Filtering Instances (Ranking-Based)...")
    masks_low_res = get_masks_coler(eigen_vec, (feat_h, feat_w), tau_filter=0.95)
    print(f"Found {len(masks_low_res)} objects.")

    # 6. Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, len(masks_low_res) + 1, 1)
    plt.imshow(img_orig)
    plt.title("Original")
    plt.axis('off')

    img_orig_np = np.array(img_orig)

    for i, mask in enumerate(masks_low_res):
        # Resize mask back to padded input size
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        
        # Using nearest neighbor to keep sharp edges (since resolution is high)
        # Or change to 'bilinear' if you prefer smoother edges
        mask_up = torch.nn.functional.interpolate(
            mask_tensor, size=(h_model, w_model), mode='nearest'
        ).squeeze().numpy()
        
        # Crop padding to match valid original image area
        mask_up = mask_up[:valid_h, :valid_w]

        # Optional CRF
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
    parser.add_argument("--tau", type=float, default=0.15) 
    parser.add_argument("--use_crf", action='store_true', help="Enable CRF (CutOnce+)")
    
    args = parser.parse_args()
    main(args)