---
title: Paper
layout: default
nav_order: 6
description: "The four-page paper (PDF) and a BibTeX entry."
---

# Paper
{: .no_toc }

*Reading Shallow Flood Depth from Real USGS Gauge-Camera Footage with a Learned Staff-Plate Reader.* Kalp Shah, IIT Gandhinagar (2026). Four pages, NeurIPS-style preprint format.

<p>
  <a class="btn btn-primary" href="{{ '/assets/paper.pdf' | relative_url }}">Download PDF</a>
  <a class="btn" href="https://github.com/kalpshah18/FloodCam/tree/main/paper">LaTeX source</a>
</p>

Every number, table and figure in the paper is regenerated from `results/` by `scripts/make_paper_assets.py`, and the numbers on this website come from the same generated file.

<object data="{{ '/assets/paper.pdf' | relative_url }}" type="application/pdf" width="100%" height="900">
  <p>Your browser cannot show the PDF inline. <a href="{{ '/assets/paper.pdf' | relative_url }}">Open it directly</a>.</p>
</object>

## Cite

```bibtex
@misc{shah2026floodcam,
  author       = {Shah, Kalp},
  title        = {Reading Shallow Flood Depth from Real {USGS} Gauge-Camera Footage
                  with a Learned Staff-Plate Reader},
  year         = {2026},
  note         = {IIT Gandhinagar},
  howpublished = {\url{https://github.com/kalpshah18/FloodCam}}
}
```

## References used in the paper

- R. Vandaele et al. River-level monitoring from camera images. *HESS* 25, 2021.
- Y. Liang et al. V-FloodNet: urban flood video segmentation. *Environmental Modelling and Software*, 2023.
- USGS NIMS images and NWIS gage height; GitHub projects `waterline` and `stagesight`.
- D. Lowe. Distinctive image features from scale-invariant keypoints (SIFT). *IJCV* 60(2), 2004.
- F. Pedregosa et al. Scikit-learn: machine learning in Python. *JMLR* 12, 2011.
