# Detecting AI-Generated Images Using 3D Cues

COMPSCI 180/280 course project. Team: Yahu Cong, Alexander Tsai, Jiankang Chang, Arjun Arin Nanduri.

## Motivation
Real photographs come from physical 3D scenes. Image generators learn from 2D images and are not forced to respect 3D structure. If so, 3D cues recovered from a single image could reveal whether the image is real or generated.

## Research question
Can 3D cues inferred from a single image tell real photographs from AI-generated images?

## Experimental setup
- **Task:** classify indoor images as real or AI-generated.
- **Splits:** by caption, so a scene never appears on both sides of a split.
- **Evaluation:** AUROC with bootstrap confidence intervals, reported per generator.
- **Baseline:** every result is shown next to an image-only baseline.
- **Generalization:** a final test on a generator that was not seen in training.

## Dataset
MS COCOAI: real MS COCO photographs and images from five generators (Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL·E 3, Midjourney v6), all generated from the captions of the real photographs.

We use only its training and validation splits, because the test split contains transformed copies of the images (flipped, half-brightness, noisy, JPEG-compressed). Indoor scenes are selected by caption (room words such as kitchen, bathroom, bedroom, living room; 7 outdoor captions removed). This gives 1,424 captions and 8,652 images, 1,442 per source (real and each generator). All images are center-cropped, resized to 256 × 256 and re-encoded as JPEG quality 95.

## Limitations
Indoor scenes are dominated by bathrooms and kitchens; the generators are from 2023–24; the scale is small; and the pretrained models that estimate 3D cues make their own mistakes.
