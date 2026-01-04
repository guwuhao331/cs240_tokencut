import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from tokencut.networks import get_model, get_coler_model
from tokencut.object_discovery import ncut as tokencut_ncut
from tokencut.object_discovery import fast_ncut_optimized as fastcut_ncut
from tokencut.cutonce import ncut_coler, get_masks_coler, densecrf_post_process


def pad_pil_to_patch_size(image: Image.Image, patch_size: int):
    w, h = image.size
    new_w = int(np.ceil(w / patch_size) * patch_size)
    new_h = int(np.ceil(h / patch_size) * patch_size)
    if new_w == w and new_h == h:
        return image, (w, h), (w, h)
    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded.paste(image, (0, 0))
    return padded, (new_w, new_h), (w, h)


def resize_to_max_box(image: Image.Image, max_side: int, min_side: int):
    """
    Downscale (never upscale) to fit within a typical resolution box.

    - max(width, height) <= max_side
    - min(width, height) <= min_side
    """
    if (max_side is None or max_side <= 0) and (min_side is None or min_side <= 0):
        return image, False

    w, h = image.size
    long_side = max(w, h)
    short_side = min(w, h)

    scale_candidates = [1.0]
    if max_side is not None and max_side > 0:
        scale_candidates.append(max_side / float(long_side))
    if min_side is not None and min_side > 0:
        scale_candidates.append(min_side / float(short_side))
    scale = min(scale_candidates)

    if scale >= 1.0:
        return image, False

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), resample=Image.BICUBIC), True


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(173, 216, 230), alpha=0.7):
    mask = mask.astype(bool)
    out = rgb.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def overlay_instances(
    rgb: np.ndarray,
    masks: list,
    colors: list,
    alpha=0.7,
):
    out = rgb.copy()
    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        mask_bool = mask.astype(bool)
        out[mask_bool] = (out[mask_bool] * (1 - alpha) + np.array(color) * alpha).astype(
            np.uint8
        )
    return out


def save_binary_png(path: Path, mask: np.ndarray):
    mask_u8 = ((mask > 0.5).astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(path)


def draw_bbox(ax, bbox, color="lime"):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)


def draw_bboxes(ax, bboxes, color="lime"):
    if not bboxes:
        return
    for bbox in bboxes:
        draw_bbox(ax, bbox, color=color)


def bbox_from_binary_mask(mask: np.ndarray):
    mask_bool = mask > 0.5
    if mask_bool.ndim != 2:
        return None
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2) + 1, int(y2) + 1


def extract_tokencut_features(model, input_tensor, which_features: str):
    feat_out = {}

    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output

    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
        hook_fn_forward_qkv
    )

    attentions = model.get_last_selfattention(input_tensor)
    qkv = feat_out["qkv"]
    bs, nb_head, nb_token = attentions.shape[0], attentions.shape[1], attentions.shape[2]
    qkv = qkv.reshape(bs, nb_token, 3, nb_head, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q.transpose(1, 2).reshape(bs, nb_token, -1)
    k = k.transpose(1, 2).reshape(bs, nb_token, -1)
    v = v.transpose(1, 2).reshape(bs, nb_token, -1)

    if which_features == "q":
        return q
    if which_features == "v":
        return v
    return k


def extract_coler_k_features(model, input_tensor):
    q, k, v = model.get_last_features(input_tensor)
    feat = k[0]  # (1, tokens, dim)
    feat = feat[:, 1:, :]  # drop CLS
    feat = feat.permute(0, 2, 1)  # (1, dim, tokens)
    return feat.reshape(-1, feat.shape[-1])  # (dim, tokens)


def cutonce_principal_mask(eigen_vec: torch.Tensor, dims):
    h, w = dims
    avg = torch.mean(eigen_vec)
    bip = (eigen_vec > avg).reshape(h, w).cpu().numpy()

    corners = int(bip[0, 0]) + int(bip[0, -1]) + int(bip[-1, 0]) + int(bip[-1, -1])
    if corners >= 3:
        bip = np.logical_not(bip)

    from scipy import ndimage

    bip = ndimage.binary_fill_holes(bip)
    objects, num_objects = ndimage.label(bip)
    if num_objects < 1:
        return np.zeros((h, w), dtype=np.uint8)

    labels, counts = np.unique(objects, return_counts=True)
    if labels[0] == 0:
        labels = labels[1:]
        counts = counts[1:]
    if len(labels) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    best = labels[np.argmax(counts)]
    return (objects == best).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Run TokenCut / FastCut / CutOnce / COLER on the same image and save a comparison figure."
    )
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/compare")

    parser.add_argument("--arch", type=str, default="vit_base", choices=["vit_small", "vit_base"])
    parser.add_argument("--patch_size", type=int, default=8, choices=[8, 16])
    parser.add_argument("--which_features", type=str, default="k", choices=["k", "q", "v"])
    parser.add_argument("--tokencut_tau", type=float, default=0.2)
    parser.add_argument("--fastcut_tau", type=float, default=0.2)

    parser.add_argument("--cutonce_tau", type=float, default=0.15)
    parser.add_argument("--tau_filter", type=float, default=0.95)
    parser.add_argument("--max_instances", type=int, default=10)
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument(
        "--resize_preset",
        type=str,
        default="480p",
        choices=["none", "480p"],
        help="Downscale large inputs before running (default: 480p-ish; helps avoid OOM).",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    out_dir = Path(args.out_dir) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_total = time.perf_counter()
    timings = {"_mode": "summary_and_method_breakdown"}

    t0_setup = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_orig = Image.open(image_path).convert("RGB")

    if args.resize_preset == "480p":
        # A "480p-like" box: within 854x480 (never upscale).
        max_side, min_side = 854, 480
    else:
        max_side, min_side = None, None

    img_orig, resized = resize_to_max_box(img_orig, max_side=max_side, min_side=min_side)
    if resized:
        print(
            f"[resize] resized input to {img_orig.size[0]}x{img_orig.size[1]} (preset={args.resize_preset})"
        )

    img_padded, (w_model, h_model), (valid_w, valid_h) = pad_pil_to_patch_size(
        img_orig, args.patch_size
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    input_tensor = transform(img_padded).unsqueeze(0).to(device)

    feat_h = h_model // args.patch_size
    feat_w = w_model // args.patch_size
    dims = (feat_h, feat_w)
    scales = (args.patch_size, args.patch_size)
    init_image_size = (3, h_model, w_model)

    rgb = np.array(img_orig)
    timings["setup_s"] = time.perf_counter() - t0_setup

    def upsample_and_crop(mask_low):
        mask_tensor = torch.from_numpy(mask_low).float().unsqueeze(0).unsqueeze(0)
        mask_up = torch.nn.functional.interpolate(
            mask_tensor, size=(h_model, w_model), mode="nearest"
        ).squeeze().numpy()
        return mask_up[:valid_h, :valid_w]

    instance_colors = [
        (255, 99, 71),   # tomato
        (60, 179, 113),  # mediumseagreen
        (65, 105, 225),  # royalblue
        (238, 130, 238), # violet
        (255, 215, 0),   # gold
        (0, 206, 209),   # darkturquoise
        (255, 140, 0),   # darkorange
        (154, 205, 50),  # yellowgreen
        (0, 191, 255),   # deepskyblue
        (221, 160, 221), # plum
    ]

    # --- TokenCut timings (load -> features -> infer -> post -> save)
    tokencut_breakdown = {}
    t0 = time.perf_counter()
    model_tokencut = get_model(args.arch, args.patch_size, device)
    tokencut_breakdown["load_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.no_grad():
        feats_tokencut = extract_tokencut_features(
            model_tokencut, input_tensor, args.which_features
        )
    tokencut_breakdown["features_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    with torch.no_grad():
        pred_tokencut, _, mask_tokencut, _, _, _ = tokencut_ncut(
            feats_tokencut,
            dims=(feat_h, feat_w),
            scales=scales,
            init_image_size=init_image_size,
            tau=args.tokencut_tau,
            eps=1e-5,
        )
    tokencut_breakdown["infer_s"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    mask_tokencut_up = upsample_and_crop(mask_tokencut.astype(np.uint8))
    tokencut_overlay = overlay_mask(rgb, mask_tokencut_up)
    tokencut_breakdown["post_s"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    Image.fromarray(tokencut_overlay).save(out_dir / "tokencut_overlay.png")
    save_binary_png(out_dir / "tokencut_mask.png", mask_tokencut_up)
    tokencut_breakdown["save_s"] = time.perf_counter() - t4

    timings["tokencut_total_s"] = sum(tokencut_breakdown.values())
    timings["tokencut_breakdown"] = tokencut_breakdown

    # --- FastCut timings (load -> features -> infer -> post -> save)
    fastcut_breakdown = {}
    t0 = time.perf_counter()
    model_fastcut = get_model(args.arch, args.patch_size, device)
    fastcut_breakdown["load_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.no_grad():
        feats_fastcut = extract_tokencut_features(
            model_fastcut, input_tensor, args.which_features
        )
    fastcut_breakdown["features_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    with torch.no_grad():
        pred_fastcut, _, mask_fastcut, _, _ = fastcut_ncut(
            feats_fastcut,
            dims=(feat_h, feat_w),
            scales=scales,
            init_image_size=init_image_size,
            tau=args.fastcut_tau,
            eps=1e-5,
        )
    fastcut_breakdown["infer_s"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    mask_fastcut_up = upsample_and_crop(mask_fastcut.astype(np.uint8))
    fastcut_overlay = overlay_mask(rgb, mask_fastcut_up)
    fastcut_breakdown["post_s"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    Image.fromarray(fastcut_overlay).save(out_dir / "fastcut_overlay.png")
    save_binary_png(out_dir / "fastcut_mask.png", mask_fastcut_up)
    fastcut_breakdown["save_s"] = time.perf_counter() - t4

    timings["fastcut_total_s"] = sum(fastcut_breakdown.values())
    timings["fastcut_breakdown"] = fastcut_breakdown

    # --- CutOnce timings (load -> features -> infer -> post -> save)
    cutonce_breakdown = {}
    t0 = time.perf_counter()
    model_cutonce = get_coler_model(arch=args.arch, patch_size=args.patch_size, device=device)
    cutonce_breakdown["load_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.no_grad():
        feats_coler = extract_coler_k_features(model_cutonce, input_tensor)
    cutonce_breakdown["features_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    with torch.no_grad():
        eigen_vec = ncut_coler(feats_coler, dims=dims, tau=args.cutonce_tau)
    cutonce_breakdown["infer_s"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    mask_cutonce = cutonce_principal_mask(eigen_vec, dims=dims)
    mask_cutonce_up = upsample_and_crop(mask_cutonce.astype(np.uint8))
    if args.use_crf:
        mask_cutonce_up = densecrf_post_process(rgb, mask_cutonce_up)
    cutonce_overlay = overlay_mask(rgb, mask_cutonce_up)
    cutonce_bbox = bbox_from_binary_mask(mask_cutonce_up)
    cutonce_breakdown["post_s"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    Image.fromarray(cutonce_overlay).save(out_dir / "cutonce_overlay.png")
    save_binary_png(out_dir / "cutonce_mask.png", mask_cutonce_up)
    cutonce_breakdown["save_s"] = time.perf_counter() - t4

    timings["cutonce_total_s"] = sum(cutonce_breakdown.values())
    timings["cutonce_breakdown"] = cutonce_breakdown

    # --- COLER timings (reuse CutOnce eigen_vec; post -> save)
    coler_breakdown = {"reuse_cutonce_eigen_vec": 1}

    t0 = time.perf_counter()
    masks_coler = get_masks_coler(eigen_vec, dims=dims, tau_filter=args.tau_filter)
    masks_coler = masks_coler[: max(0, args.max_instances)]
    masks_coler_up = [upsample_and_crop(m.astype(np.uint8)) for m in masks_coler]

    union_coler_up = (
        np.maximum.reduce(masks_coler_up) if len(masks_coler_up) else np.zeros((valid_h, valid_w))
    )
    if args.use_crf:
        union_coler_up = densecrf_post_process(rgb, union_coler_up)

    coler_overlay = overlay_instances(rgb, masks_coler_up, instance_colors)
    coler_bboxes = [bbox_from_binary_mask(m_up) for m_up in masks_coler_up]
    coler_bboxes = [b for b in coler_bboxes if b is not None]
    coler_breakdown["post_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    Image.fromarray(coler_overlay).save(out_dir / "coler_overlay.png")
    save_binary_png(out_dir / "coler_mask.png", union_coler_up)
    for idx, m_up in enumerate(masks_coler_up, start=1):
        save_binary_png(out_dir / f"coler_instance_{idx:02d}.png", m_up)
    coler_breakdown["save_s"] = time.perf_counter() - t1

    # For comparison, define COLER as "CutOnce pre-steps (load+features+infer) + COLER post+save".
    # This reflects that COLER relies on the CutOnce eigen-vector as input.
    coler_breakdown["pre_s"] = float(cutonce_breakdown.get("load_s", 0.0)) + float(
        cutonce_breakdown.get("features_s", 0.0)
    ) + float(cutonce_breakdown.get("infer_s", 0.0))
    timings["coler_total_s"] = float(coler_breakdown.get("pre_s", 0.0)) + float(
        coler_breakdown.get("post_s", 0.0)
    ) + float(coler_breakdown.get("save_s", 0.0))
    timings["coler_breakdown"] = coler_breakdown

    panels = [
        ("Original", rgb, None),
        ("TokenCut", tokencut_overlay, pred_tokencut),
        ("FastCut", fastcut_overlay, pred_fastcut),
        ("CutOnce", cutonce_overlay, cutonce_bbox),
        ("COLER (multi)", coler_overlay, coler_bboxes),
    ]

    fig = plt.figure(figsize=(20, 4))
    for i, (title, img_panel, bbox) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), i)
        ax.imshow(img_panel)
        ax.set_title(title)
        ax.axis("off")
        if bbox is not None:
            if isinstance(bbox, list):
                draw_bboxes(ax, bbox)
            else:
                draw_bbox(ax, bbox)

    out_png = out_dir / "compare.png"
    t0 = time.perf_counter()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    timings["compare_png_s"] = time.perf_counter() - t0

    timings["total_s"] = time.perf_counter() - t0_total

    def _fmt(seconds: float) -> str:
        return f"{seconds:.3f}s"

    print("\nTimings:")
    print(f"  total: {_fmt(timings['total_s'])}")
    print(f"  tokencut: {_fmt(timings['tokencut_total_s'])}")
    print(f"  fastcut: {_fmt(timings['fastcut_total_s'])}")
    print(f"  cutonce: {_fmt(timings['cutonce_total_s'])}")
    print(f"  coler: {_fmt(timings['coler_total_s'])}")

    with open(out_dir / "timings.txt", "w", encoding="utf-8") as f:
        f.write(f"total\t{timings['total_s']:.6f}\n")
        f.write(f"tokencut\t{timings['tokencut_total_s']:.6f}\n")
        f.write(f"fastcut\t{timings['fastcut_total_s']:.6f}\n")
        f.write(f"cutonce\t{timings['cutonce_total_s']:.6f}\n")
        f.write(f"coler\t{timings['coler_total_s']:.6f}\n")
        f.write("\n")
        f.write(f"setup_s\t{timings['setup_s']:.6f}\n")
        f.write(f"compare_png_s\t{timings['compare_png_s']:.6f}\n")
        for name in ["tokencut", "fastcut", "cutonce", "coler"]:
            b = timings.get(f"{name}_breakdown", {})
            if not b:
                continue
            for kk in sorted(b.keys()):
                if isinstance(b[kk], (int, float)):
                    f.write(f"{name}.{kk}\t{float(b[kk]):.6f}\n")
                else:
                    f.write(f"{name}.{kk}\t{b[kk]}\n")

    print(f"Saved: {out_png}")
    print(f"Saved masks in: {out_dir}")


if __name__ == "__main__":
    main()
