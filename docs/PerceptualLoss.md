# Perceptual loss

OneTrainer now has an optional `perceptual_loss` config block for x0-decoded auxiliary losses. The hook reconstructs the model's current clean latent prediction from the existing diffusion or flow prediction, decodes that latent through the training VAE, and adds optional image-space losses to the normal training objective.

The default config is disabled and all weights default to `0.0`, so existing presets keep their previous behavior.

## Supported setup classes

The hook is wired into the image setup classes that expose a VAE-decodable x0 latent:

- Stable Diffusion / Stable Diffusion XL
- PixArt Alpha
- Flux / Flux 2
- Chroma
- Stable Diffusion 3
- HiDream
- Sana
- Z-Image
- Ernie
- Qwen image latents with a singleton frame dimension

The hook intentionally does not apply to Wuerstchen/Stable Cascade prior training or Hunyuan video latents.

## Config

Add the block to a training preset or set the same values in the Training tab.

```json
"perceptual_loss": {
  "enabled": true,
  "min_t": 0.0,
  "max_t": 1.0,
  "decode_chunk_size": 1,

  "decoded_l1_weight": 0.0,
  "decoded_mse_weight": 0.0,
  "edge_weight": 0.0,

  "depth_weight": 0.05,
  "depth_model_id": "depth-anything/Depth-Anything-V2-Small-hf",
  "depth_input_size": 518,
  "depth_dtype": "bfloat16",
  "depth_grad_checkpoint": true,
  "depth_ssi_weight": 1.0,
  "depth_grad_weight": 0.5,
  "depth_grad_scales": 4,
  "depth_pixel_blur_sigma": 0.0
}
```

## Losses

- `decoded_l1_weight`: pixel-space L1 between decoded predicted x0 and decoded target latents.
- `decoded_mse_weight`: pixel-space MSE between decoded predicted x0 and decoded target latents.
- `edge_weight`: Sobel edge L1 between decoded predicted x0 and decoded target latents.
- `depth_weight`: Depth-Anything-V2 scale/shift-invariant depth consistency, adapted from `ai-toolkit-perceptual`'s differentiable depth-consistency path.

The depth target is computed from the decoded target latent during training. This avoids adding a new cache format and keeps the target aligned with the exact VAE representation OneTrainer can produce.

## Notes

- `min_t` and `max_t` gate the normalized diffusion timestep range where perceptual loss is active. The default `min_t: 0.0` and `max_t: 1.0` apply it across the full schedule; set `max_t: 0.3` for early denoising only, or `min_t: 0.7` for late refinement only.
- Perceptual loss decodes through the VAE during the training step. Expect extra VRAM and runtime cost.
- `decode_chunk_size` lowers peak VRAM by decoding fewer samples at once.
- Prior-preservation samples are excluded from the perceptual auxiliary loss when `concept_type` identifies them as prior prediction samples.
- TensorBoard logs are emitted under `loss/perceptual*` when a perceptual component is active.
