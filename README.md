# TG-SEG

## Text-Guided and Presence-Aware Temporal Propagation for Medical Image Segmentation

**Accepted at CIKM 2026**

**Krishna Tewari\***, **Shayon Dasgupta\***, and **Sukomal Pal**  
Indian Institute of Technology (BHU), Varanasi, India  
\* Equal contribution.

[Paper](https://doi.org/10.1145/3799682.3840063) | [Repository](https://github.com/cse-iitbhu/TG-SEG)

## Overview

TG-SEG combines text-guided medical image segmentation with temporal propagation and presence-aware gating. It handles absent targets by returning empty masks and resetting propagation state, helping prevent temporal drift across frames or slices.

## Repository Structure

| Folder | Description |
|---|---|
| `tgseg/src/` | Core model, visual and text encoders, temporal propagation, training, evaluation, and utilities. |
| `tgseg/ablation/` | Configurations and scripts for encoder comparisons and presence-aware ablation experiments. |

## Acknowledgments

TG-SEG builds on the architecture and temporal propagation design of [Text-Promptable Propagation for Referring Medical Image Sequence Segmentation (TPP)](https://doi.org/10.1145/3746027.3755166).

We thank the authors of the datasets and baseline methods used in this study.

## Citation

If you use TG-SEG in your research, please cite:

```bibtex
@inproceedings{tewari2026tgseg,
  author    = {Tewari, Krishna and Dasgupta, Shayon and Pal, Sukomal},
  title     = {{TG-SEG}: Text-Guided and Presence-Aware Temporal Propagation
               for Medical Image Segmentation},
  booktitle = {Proceedings of the 35th ACM International Conference
               on Information and Knowledge Management},
  series    = {CIKM '26},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  location  = {Rome, Italy},
  doi       = {10.1145/3799682.3840063},
  url       = {https://doi.org/10.1145/3799682.3840063}
}
```

## Contact

- **Krishna Tewari:** [krishnatewari.rs.cse24@itbhu.ac.in](mailto:krishnatewari.rs.cse24@itbhu.ac.in)
- **Shayon Dasgupta:** [shayon.dasgupta.cse23@itbhu.ac.in](mailto:shayon.dasgupta.cse23@itbhu.ac.in)
