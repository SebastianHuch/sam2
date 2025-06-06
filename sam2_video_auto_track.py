import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2_auto_mask import show_anns

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def main():
    parser = argparse.ArgumentParser(description="SAM2 Video: Segment and Track All Objects (No Prompt)")
    parser.add_argument('--video-dir', required=True, help='Directory with JPEG video frames (00000.jpg, ...)')
    parser.add_argument('--checkpoint', required=True, help='Path to SAM2 model checkpoint')
    parser.add_argument('--config', required=True, help='Path to SAM2 model config YAML')
    parser.add_argument('--output-dir', default=None, help='Output dir for visualizations (optional)')
    parser.add_argument('--stride', type=int, default=30, help='Visualize every N frames')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot windows')
    parser.add_argument('--max-pixels', type=int, default=500000, help='Max pixels to downsample initial frame')
    # You can add more maskgen options if you want (points_per_side, thresholds, etc.)
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Frame listing
    frame_names = [p for p in os.listdir(args.video_dir)
                   if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    print(f"Found {len(frame_names)} frames.")

    # Load first frame, downsample if needed
    def downsample_to_max_pixels(image_pil, max_pixels):
        w, h = image_pil.size
        num_pixels = w * h
        if num_pixels > max_pixels:
            scale = (max_pixels / num_pixels) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            print(f"Downsampling from {w}x{h} ({num_pixels} px) to {new_w}x{new_h} ({new_w*new_h} px)")
            return image_pil.resize((new_w, new_h), Image.LANCZOS)
        return image_pil

    first_img = Image.open(os.path.join(args.video_dir, frame_names[0])).convert("RGB")
    first_img = downsample_to_max_pixels(first_img, args.max_pixels)
    first_img_np = np.array(first_img)

    # --- Step 1: Generate masks on first frame ---
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
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
    print("Generating automatic masks on first frame...")
    masks = mask_generator.generate(first_img_np)
    print(f"Generated {len(masks)} masks for frame 0.")

    # Visualize the initial masks
    plt.figure(figsize=(10, 8))
    plt.title("Initial automatic masks (frame 0)")
    plt.imshow(first_img_np)
    # add obj_id to each mask for visualization
    for i, mask in enumerate(masks):
        mask['obj_id'] = i
    show_anns(masks)
    plt.axis('off')
    plt.tight_layout()

    if not args.no_show:
        plt.show()

    # --- Step 2: Build video predictor ---
    predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=device)
    inference_state = predictor.init_state(video_path=args.video_dir)

    # --- Step 3: Add all masklets (no prompt, just initial masks) ---
    # Use unique obj_ids for each mask (e.g., 1, 2, 3, ...)
    for i, mask in enumerate(masks):
        # mask['segmentation'] is (H, W) boolean/numpy
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=i + 1,
            mask=mask['segmentation'].astype(np.uint8)
        )

    # --- Step 4: Propagate all segments across video ---
    print("Propagating all initial masks across video (this may take time)...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # --- Step 5: Visualize results every N frames ---
    vis_stride = args.stride
    for out_frame_idx in range(0, len(frame_names), vis_stride):
        plt.figure(figsize=(10, 8))
        plt.title(f"Tracked masks: frame {out_frame_idx}")
        img = Image.open(os.path.join(args.video_dir, frame_names[out_frame_idx]))
        plt.imshow(img)
        annos = []
        for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
            annos.append({
                'segmentation': out_mask[0] if out_mask.ndim == 3 else out_mask,
                'area': 0,  # area is not used in visualization
                'obj_id': out_obj_id
            })
        show_anns(annos)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            plt.savefig(os.path.join(args.output_dir, f"tracked_masks_frame_{out_frame_idx}.png"))
        # if not args.no_show:
        #     plt.show()

if __name__ == "__main__":
    main()
