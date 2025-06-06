import argparse
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

def get_unique_color(obj_id, alpha=0.5, cmap_name="tab20"):
    cmap = matplotlib.cm.get_cmap(cmap_name)
    # Ensure obj_id is always an int and in colormap range
    color = np.array(cmap(obj_id % cmap.N))  # RGBA, last channel is alpha
    color[3] = alpha
    return color

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    # Area is not relevant for sorting in this use-case, but sort to avoid color bleeding
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((*anns[0]['segmentation'].shape, 4))
    img[:, :, 3] = 0
    for ann in anns:
        m = ann['segmentation']
        obj_id = ann.get('obj_id', 0)
        color_mask = get_unique_color(obj_id)
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ax.imshow(img)

def downsample_to_max_pixels(image_pil, max_pixels=500_000):
    w, h = image_pil.size
    num_pixels = w * h
    if num_pixels > max_pixels:
        scale = (max_pixels / num_pixels) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Downsampling from {w}x{h} ({num_pixels} px) to {new_w}x{new_h} ({new_w*new_h} px)")
        return image_pil.resize((new_w, new_h), Image.LANCZOS)
    return image_pil

def main():
    parser = argparse.ArgumentParser(description="SAM2 Automatic Mask Generator")
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--checkpoint', required=True, help='Path to SAM2 model checkpoint')
    parser.add_argument('--config', required=True, help='Path to SAM2 model config YAML')
    parser.add_argument('--output', default=None, help='Output PNG file to save mask visualization (optional)')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot (for headless systems)')
    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # CUDA precision options
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Load image
    image = Image.open(args.image).convert("RGB")
    image = downsample_to_max_pixels(image, max_pixels=500_000)
    image_np = np.array(image)

    # Build model and mask generator
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2 = build_sam2(args.config, args.checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.94,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.4,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)

    # Generate masks
    masks = mask_generator.generate(image_np)
    print(f"Generated {len(masks)} masks")

    # Plot and optionally save/show
    plt.figure(figsize=(16, 16))
    plt.imshow(image_np)
    show_anns(masks)
    plt.axis('off')
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
        print(f"Saved visualization to {args.output}")

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()

