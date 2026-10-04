# Implementation cookbook (verified against source, 2026-09-28)

Condensed from a source-level read of pycolmap 3.12.6 / 4.2.0, DepthAnything/PromptDA, and
facebookresearch/sam3 (incl. RELEASE_SAM3p1.md). Full details live in the code under
`pipeline/foodscan/{refine_poses,depth_upsample,segment}.py`. Runtimes are estimates.

## pycolmap known-pose refinement
- Pins: `pycolmap==3.12.6` (frames/rigs model) or 4.2.0 (adds `add_image_with_trivial_frame`,
  GLOMAP `global_mapping`). PyPI wheels are CPU SIFT; `pip install pycolmap-cuda12` (≥4.0.4, Linux) for GPU.
- Conventions: `Rotation3d` is **[x,y,z,w]**; `Rigid3d` is always **cam_from_world**.
  ARKit c2w → COLMAP: `T_cw = inv(T_c2w_arkit @ diag(1,-1,-1,1))`.
- `triangulate_points` **never moves poses** (hard-coded `fix_existing_frames=True`, issue #3433).
  Pose refinement = separate `bundle_adjustment()`; it fixes gauge with two cameras only, so the
  result drifts in scale/rotation (issue #595) and must be re-aligned to the ARKit frame.
- `BundleAdjustmentOptions` 3.12: `refine_focal_length/principal_point/extra_params=False`,
  `refine_rig_from_world=True`, `refine_sensor_from_rig=False`. (3.11 uses `refine_extrinsics`.)
- Re-align with `align_reconstructions_via_proj_centers(rec, ref, max_proj_center_error)`; assert
  the scale comes back ≈1 (metric should be preserved), or do a strict SE3 Umeyama.
- Pose priors (`Database.write_pose_prior`) are position-only: soft anchor, not a substitute.
- GLOMAP (4.x `global_mapping`) ignores known poses.
- Pitfalls: needs ≥2 registered images and the `image_path` present; names must match the DB;
  3.12 `image.cam_from_world` is a **method**; `register_image` → `register_frame`.
  Tiny video baselines → subsample every ~3rd frame for matching or raise `triangulation.min_angle`.
- Runtime 450 frames, CPU SIFT 8k, overlap 10: ~2–4 min extract, 3–6 min match, 1–3 min tri+BA.

## PromptDA
- `requirements.txt`: python 3.9–3.11, torch 2.0.1 (newer works in practice), xformers optional
  (falls back to plain attention), **open3d needed** (imported by `depth_utils`). `pip install -e .`
  installs nothing → run from repo root / PYTHONPATH.
- `PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl")` (1.36 GB; vits needs
  `model_kwargs={"encoder": "vits"}`).
- Contract: `image` float32 (1,3,H,W) in **[0,1]** (normalisation inside), **H,W multiples of 14**
  (no auto-pad, crashes otherwise) → use 1904×1428 and resize back; `prompt_depth` float32 (1,1,h,w)
  **metres**, any resolution; output (1,1,H,W) metres.
- **Holes are not handled**: a 0 in the prompt sets min to 0 and hurts the CNN branch. Author's
  workaround: KNN completion (k=4) of invalid pixels before prompting. No confidence masking in repo.
- HF alternative `depth-anything/prompt-depth-anything-vitl-hf`: pass `prompt_scale_to_meter=1.0`
  (default 0.001 assumes mm).
- Speed: paper 20 FPS at 768×1024 on A100; ~1–2 min for 450 frames at 1904×1428 on L40S, <12 GB.

## SAM 3 / 3.1
- Install: python ≥3.12, torch ≥2.7, CUDA ≥12.6; `pip install -e ".[notebooks]"` + `psutil`
  (undeclared) + `"setuptools<81"` (pkg_resources). Weights gated: `facebook/sam3` (`sam3.pt`),
  `facebook/sam3.1` (`sam3.1_multiplex.pt`); builders auto-download after `hf auth login`.
  Linux only (video predictor spawns NCCL workers even on 1 GPU).
- Builders: `build_sam3_video_predictor(gpus_to_use=[0])` (SAM 3),
  `build_sam3_multiplex_video_predictor(max_num_objects=16, ...)` (SAM 3.1),
  `build_sam3_image_model()` + `Sam3Processor`.
- Requests: `start_session{resource_path: mp4|dir|list[PIL], offload_video_to_cpu}` →
  `add_prompt{frame_index, text, output_prob_thresh}` → stream `propagate_in_video{propagation_direction,
  start_frame_index, evict_cached_frame_outputs}` → `reset_session` / `close_session`.
- Per-frame outputs: `out_obj_ids` (N,), `out_probs` (N,) = frame-0 detection score,
  `out_boxes_xywh` (N,4) normalised, `out_binary_masks` bool (N,H,W) at video res, non-overlapping.
  IDs monotonic, never reused, consistent across one propagation.
- **One text concept per propagation** (text prompt resets state; issue #206/#267): loop concepts
  with `reset_session`. No negative text; negative points/boxes exist.
- Frames dir: sorted by `int(stem)`, resized to 1008², fp16 on GPU unless offloaded.
- Memory: use `offload_video_to_cpu=True`, `evict_cached_frame_outputs=True`, forward-only.
- SAM 3.1 open issue #544: `start_session` passes `offload_state_to_cpu` which multiplex lacks.
- HF transformers ≥5.0 `Sam3VideoModel` supports **multi-concept in one pass** (`facebook/sam3`
  weights only): `proc.add_text_prompt(session, ["rice", "chicken"])` →
  `model.propagate_in_video_iterator(...)` → `proc.postprocess_outputs(...)`.
- Runtime: ~30–60 s per concept for 450 frames on L40S [unverified].
