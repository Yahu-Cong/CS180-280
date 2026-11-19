# Depth-rectified texture grids

See [the report](REPORT.md) for the results and [the protocol](PROTOCOL.md) for the sample, features, and evaluation. The experiment uses the course project's indoor MS COCOAI archive and keeps the original images out of Git.

## Reproduce on Colab

Start a GPU runtime. The pilot used a Tesla T4 and the ViT-S/14 UniDepth V2 checkpoint. In a Colab terminal or a `%%bash` cell, run the commands below. Set the working directory to this directory in a checkout of the branch before running them.

```bash
set -eu
python -m pip install timm gdown
git clone https://github.com/lpiccinelli-eth/UniDepth.git /content/UniDepth
git -C /content/UniDepth checkout --detach 8d8cfe4c7ee15297099983607febf0d4f32eb3d6
python -m pip install --no-deps /content/UniDepth
python -m gdown 1WQQ7K_dboYJtw_quGQwW4On5WA3djk5w -O /content/coco_indoor_all.zip
sha256sum /content/coco_indoor_all.zip
python synthetic_check.py
python select_sample.py /content/coco_indoor_all.zip /content/texture_sample
python -u predict_depth.py /content/texture_sample /content/texture_depth --device cuda
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python -u evaluate.py /content/texture_sample /content/texture_depth /content/texture_results
```

The archive SHA-256 is `c847a0655de51d16724852da1dbe155b0998b8941f80c8a47b2577b8dc71a089`. The commands use Colab's PyTorch, torchvision, NumPy, Pillow, OpenCV, scikit-learn, matplotlib, and related scientific packages. Install the missing packages if using an environment without Colab's preinstalled stack. The model loader pins the Hugging Face revision. The CPU evaluation can be rerun from the image sample and saved depth/intrinsics arrays.

[metrics.json](results/metrics.json) contains the full held-out evaluation, paired intervals, and unseen-generator results. [scores.csv](results/scores.csv) contains per-image feature values, caption IDs, and held-out decision scores. A blank texture field means the required grid was missing. The counts record how many of the 13 windows met each condition.
