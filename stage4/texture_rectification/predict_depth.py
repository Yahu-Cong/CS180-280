"""Cache UniDepth V2 axial depth and pinhole intrinsics in input coordinates."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from unidepth.models import UniDepthV2

MODEL = "lpiccinelli/unidepth-v2-vits14"
REVISION = "038c238f06c87b6c2f5b3749fd51fbf442b1f218"


def predict(sample, output, device, batch_size):
    with (sample / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        print(f"GPU {torch.cuda.get_device_name()}; torch {torch.__version__}", flush=True)
    model = UniDepthV2.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    model.resolution_level = 0
    start = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            arrays = []
            for row in batch:
                with Image.open(sample / row["image"]) as image:
                    arrays.append(np.array(image.convert("RGB")))
            rgb = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
            result = model.infer(rgb)
            depth = result["depth"][:, 0].float().cpu().numpy()
            cameras = result["intrinsics"].float().cpu().numpy()
            for row, z, K in zip(batch, depth, cameras):
                dest = output / f"{row['source']}_{row['caption_id']}.npz"
                np.savez_compressed(dest, depth=z.astype(np.float16), intrinsics=K)
            done = offset + len(batch)
            if done == len(rows) or done % 40 == 0:
                print(f"Depth {done}/{len(rows)} in {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    predict(args.sample, args.output, args.device, args.batch_size)
