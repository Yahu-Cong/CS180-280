# Detecting AI-Generated Images Using 3D Cues

COMPSCI 180/280 course project. Team: Yahu Cong, Alexander Tsai, Jiankang Chang, Arjun Arin Nanduri.

## Motivation
Real photographs come from physical 3D scenes. Image generators learn from 2D images and are not forced to respect 3D structure. If so, the 3D structure of a scene, recovered from a single image, could reveal whether the image is real or generated.

## Research question
Can 3D cues inferred from a single image tell real photographs from AI-generated images?

## Core idea
Estimate two independent descriptions of the same surfaces: the surface orientation derived from predicted depth and camera parameters, and the surface orientation predicted directly from the image. In a real scene both describe the same physical surface and should agree. Where they disagree (the misalignment), the image may not be consistent with a real 3D scene.

## Approach
1. **Extract 3D cue maps** with pretrained models: metric depth, camera parameters and surface normals.
2. **Compare the cues** into a disagreement (misalignment) map. Edges and corners are handled separately, because the estimators are unreliable there.
3. **Train a small CNN** on the cue maps to classify real versus generated, and compare it with the same network on the plain image to see what the 3D cues add.

## Data
MS COCOAI: real MS COCO photographs and images from five generators (Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL·E 3, Midjourney v6), all generated from the real photographs' captions. We use only its training and validation splits (the test split contains transformed copies) and keep indoor scenes by caption (room words such as kitchen, bathroom, bedroom, living room, with 7 outdoor captions removed): 1,424 captions, 8,652 images, 1,442 per source. All images are center-cropped, resized to 256 × 256 and re-encoded as JPEG quality 95.

## What we have learned so far
- Averaging the misalignment over an image does not separate real from generated images, so the signal is not in how large the misalignment is on average.
- A small CNN trained on the 3D cue maps, which include the misalignment between the depth-derived and the predicted surface orientation, detects generated images well on held-out captions (AUROC 0.87, and all five generators above chance, from 0.76 to 0.95). So the 3D maps carry a meaningful, learnable signal that sits in their spatial pattern, which a simple average misses.
- For comparison, the same network on the plain image reaches 0.95, so the 3D signal is real but does not yet beat appearance. The next steps are a test on a generator never seen in training, and an ablation using only the misalignment map, to show how much of the signal comes from the misalignment itself.

## How we evaluate
Split by caption, so a scene never appears on both sides of a split. Report AUROC with confidence intervals and per-generator results, always next to an image-only baseline.

## Limitations
Indoor scenes dominated by bathrooms and kitchens; generators from 2023–24; small scale; the pretrained estimators make their own mistakes, so a high disagreement may reflect an estimator error and not an impossible scene.
