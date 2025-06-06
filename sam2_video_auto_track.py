import argparse
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2_auto_mask import show_anns

def iou(mask1, mask2):
    # Both are (H,W) boolean arrays
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def make_batch_tempdir(video_dir, batch_frame_names, temp_dir):
    # Clean temp dir if it exists, then create it
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    for i, fname in enumerate(batch_frame_names):
        src = os.path.join(video_dir, fname)
        dst = os.path.join(temp_dir, f"{i:05d}.jpg")
        try:
            os.symlink(os.path.abspath(src), dst)
        except AttributeError:
            # Windows: no symlink, fallback to copy
            shutil.copy2(src, dst)

def overlay_masks_on_image(img_np, annos, alpha=0.5, cmap_name='tab20'):
    import matplotlib
    img = img_np.copy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    img = img[..., :3]  # Make sure it's RGB
    overlay = np.zeros_like(img, dtype=np.float32)
    cmap = matplotlib.cm.get_cmap(cmap_name)
    for ann in annos:
        m = ann['segmentation']
        obj_id = ann.get('obj_id', 0)
        color = np.array(cmap(obj_id % cmap.N)[:3]) * 255
        mask = m.astype(bool)
        for c in range(3):
            overlay[..., c][mask] = color[c]
    # Blend overlays
    blended = img.astype(np.float32) * (1 - alpha) + overlay * alpha
    return blended.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="SAM2 Video: Chunked Batch Tracking")
    parser.add_argument('--video-dir', required=True, help='Directory with JPEG video frames (00000.jpg, ...)')
    parser.add_argument('--checkpoint', required=True, help='Path to SAM2 model checkpoint')
    parser.add_argument('--config', required=True, help='Path to SAM2 model config YAML')
    parser.add_argument('--output-dir', default=None, help='Output dir for visualizations (optional)')
    parser.add_argument('--batch-size', type=int, default=10, help='How many frames per batch')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot windows')
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
    num_frames = len(frame_names)
    print(f"Found {num_frames} frames.")

    # --- Build mask generator (once, reuse) ---
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

    global_obj_id_counter = 1
    global_obj_masks = {}  # obj_id -> mask (boolean array) for last frame in prev batch

    batch_starts = [0]
    while batch_starts[-1] + args.batch_size < num_frames:
        batch_starts.append(batch_starts[-1] + args.batch_size - 1)  # overlap last frame

    batch_ranges = [
        (start, min(start + args.batch_size, num_frames))
        for start in batch_starts
    ]

    for batch_idx, (batch_start, batch_end) in enumerate(batch_ranges):
        batch_frames = frame_names[batch_start:batch_end]
        print(f"\n=== Processing batch {batch_idx+1} / {len(batch_ranges)}: frames {batch_start} to {batch_end-1} ===")

        # 1. If first batch: run auto mask on batch[0]; else, use global_obj_masks as starting state
        batch_first_img = Image.open(os.path.join(args.video_dir, batch_frames[0])).convert("RGB")
        batch_first_np = np.array(batch_first_img)

        # Always run automask on the first frame of the batch
        print("Generating masks on first frame of this batch...")
        new_masks = mask_generator.generate(batch_first_np)
        print(f"Generated {len(new_masks)} candidate masks.")

        carryover_obj_ids = list(global_obj_masks.keys())
        carryover_masks = [global_obj_masks[obj_id] for obj_id in carryover_obj_ids]

        unique_new_masks = []
        for mask in new_masks:
            this_mask = mask['segmentation']
            max_iou = 0
            for old_mask in carryover_masks:
                max_iou = max(max_iou, iou(this_mask, old_mask))
            if max_iou < 0.5:
                mask['obj_id'] = global_obj_id_counter
                global_obj_masks[global_obj_id_counter] = this_mask
                global_obj_id_counter += 1
                unique_new_masks.append(mask)
        print(f"Added {len(unique_new_masks)} truly new masks in this batch.")

        # For visualization and for the batch predictor, use all carryover + new masks
        batch_new_masks = []
        for obj_id in carryover_obj_ids:
            batch_new_masks.append({'segmentation': global_obj_masks[obj_id], 'obj_id': obj_id})
        batch_new_masks += unique_new_masks

        # Visualize initial batch masks
        plt.figure(figsize=(10, 8))
        plt.title(f"Initial masks for batch {batch_idx+1} (frame {batch_start})")
        plt.imshow(batch_first_np)
        show_anns(batch_new_masks)
        plt.axis('off')
        plt.tight_layout()
        if not args.no_show:
            plt.show()
        else:
            plt.close()

        # 2. Build video predictor for this batch, init state
        temp_dir = "./tmp_batch_frames"
        make_batch_tempdir(args.video_dir, batch_frames, temp_dir)
        predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=device)
        inference_state = predictor.init_state(video_path=temp_dir)

        # 3. Add all carried and new masks to predictor as initial state for this batch
        for mask in batch_new_masks:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=mask['obj_id'],
                mask=mask['segmentation'].astype(np.uint8)
            )

        # 4. Propagate all masks for batch frames
        print(f"Propagating masks for batch {batch_idx+1} ...")
        video_segments = {}
        for rel_frame_idx, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(
                predictor.propagate_in_video(inference_state)):
            true_frame_idx = batch_start + rel_frame_idx
            video_segments[true_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 5. Update global_obj_masks for next batch (save last frame masks)
        last_frame_idx = batch_end - 1
        if last_frame_idx in video_segments:
            for obj_id, mask in video_segments[last_frame_idx].items():
                global_obj_masks[obj_id] = mask[0] if mask.ndim == 3 else mask

        # 6. Visualize results every N frames in this batch
        for out_frame_idx in range(batch_start, batch_end, 1):
            img_path = os.path.join(args.video_dir, frame_names[out_frame_idx])
            img_np = np.array(Image.open(img_path).convert("RGB"))
            annos = []
            for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
                annos.append({
                    'segmentation': out_mask[0] if out_mask.ndim == 3 else out_mask,
                    'area': 0,
                    'obj_id': out_obj_id
                })

            # Save overlayed image as PNG/JPG
            out_img = overlay_masks_on_image(img_np, annos, alpha=0.5)
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                out_path = os.path.join(args.output_dir, f"tracked_masks_frame_{out_frame_idx}.png")
                Image.fromarray(out_img).save(out_path)

            # Only visualize if not --no-show
            if not args.no_show:
                plt.figure(figsize=(10, 8))
                plt.title(f"Tracked masks: frame {out_frame_idx}")
                plt.imshow(out_img)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            # No figure created otherwise
        
        # Cleanup
        del predictor
        del inference_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
