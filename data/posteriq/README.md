# PosterIQ: images and overall design scores

A compact dataset of **219 posters**, copied from [creative-graphic-design/PosterIQ](https://huggingface.co/datasets/creative-graphic-design/PosterIQ), configuration `overall_rating`, split `test`.

## Files

- `posters.csv`: one row per poster, with columns `id`, `image_path`, and `overall_design_score`.
- `images/`: the 219 original source image files, without image transformations.
- `source_info.json`: source and extraction metadata.
- `SHA256SUMS`: checksums for verifying the data files.

Image paths in `posters.csv` are relative to the directory containing the CSV. Scores are numeric `gt_json` values copied without scaling. The source rubric is 1–10; this bundle's observed scores range from 2.0 to 8.8.

## Load in Python

Requires Pillow (`pip install pillow`). Run from the repository root:

```python
import csv
from pathlib import Path
from PIL import Image

root = Path("data/posteriq")
with (root / "posters.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

row = rows[0]
with Image.open(root / row["image_path"]) as image:
    image = image.convert("RGB")
score = float(row["overall_design_score"])
print(row["id"], image.size, score)
```

These labels measure **overall design quality**, not whether a poster needs more text. Text quantity or text-need labels must be added separately. This source is a benchmark **test** split; after using these examples for training or model selection, do not treat them as a held-out test set.

## Source and usage

Source revision: `02bddb802658cb952d693d289af230644401bb26`.

Attribution: creative-graphic-design/PosterIQ. The source dataset card lists `non-commercial-research-license`; its LICENSE file was empty when the card was created. Consult the [source repository](https://huggingface.co/datasets/creative-graphic-design/PosterIQ/tree/02bddb802658cb952d693d289af230644401bb26) for the publisher's license and usage information. This bundle does not grant additional rights.
