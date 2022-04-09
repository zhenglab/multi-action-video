# Model Zoo

## M-MiT V1

| model | backbone | top1 | top5 | mAP | model | M3A configs |
| :---: | :------: | :--: | :--: | :-: | :---: | :---------: |
| {ν} | R34 | 58.7 | 83.7 | 61.7 | [`link`](https://drive.google.com/file/d/1ieR_2NpWhYCtz_AUezC80MoLEAyyMAqa/view?usp=sharing) | NONE,NONE,NONE |
| **G**-{ν,α} | R34 | 60.4 | 85.3 | 64.7 | [`link`](https://drive.google.com/file/d/1t45EvgSmMASm5O4QpGKGd_GZEic4ZnMR/view?usp=sharing) | GRAPH,AUDIO,NONE |
| **G**-{ν,τ} | R34 | 60.5 | 85.5 | 64.4 | [`link`](https://drive.google.com/file/d/1UHawdwx8OGQtb02mlCaY_P9VKrpmeR5k/view?usp=sharing) | GRAPH,TEXT,NONE |
| **G**-{ν,α,τ} | R34 | 60.7 | 85.6 | 65.0 | [`link`](https://drive.google.com/file/d/1S1RcJTay-jg1p6WTqsatWPpU5Ip1gg-F/view?usp=sharing) | GRAPH,AUDIO_TEXT,SUM |
| **T**-{ν,α} | R34 | 61.0 | 85.4 | 65.0 | [`link`](https://drive.google.com/file/d/1I-BfqGg2Y_O2knN4Moe6z9hc6FWUbNc9/view?usp=sharing) | TRSFMR,AUDIO,NONE |
| **T**-{ν,τ} | R34 | 61.1 | 85.8 | 65.4 | [`link`](https://drive.google.com/file/d/1loQb_PO6PXa35hKeLI1OHki-gjk5PX9M/view?usp=sharing) | TRSFMR,TEXT,NONE |
| **T**-{ν,α,τ} | R34 | 61.6 | 85.5 | 65.9 | [`link`](https://drive.google.com/file/d/1MkvxHYXxPKg7lZXENnvG0Cs3-2L9-OdS/view?usp=sharing) | TRSFMR,AUDIOTEXT,NONE |

Notes:
- The "M3A configs" corresponds to "M3A_MODE,M3A_MODAL_TYPE,M3A_MODAL_JOINT_TYPE" in [scripts/run_mmit_r2plus1d.sh](scripts/run_mmit_r2plus1d.sh).
- The baseline "{ν}" model is trained by [configs/Mmit/R2PLUS1D_GRAPH_8x2.yaml](configs/Mmit/R2PLUS1D_GRAPH_8x2.yaml).

## M-MiT V2

| model | backbone | top1 | top5 | mAP | model | M3A configs |
| :---: | :------: | :--: | :--: | :-: | :---: | :---------: |
| {ν} | R34 | 59.2 | 84.4 | 62.5 | [`link`](https://drive.google.com/file/d/18hXHxAzBPb4CSozaH4u-5Q42ea-fTpn7/view?usp=sharing) | NONE,NONE,NONE |
| **G**-{ν,α} | R34 | 61.2 | 85.9 | 65.2 | [`link`](https://drive.google.com/file/d/1tMwNucWzTC05_SapB6Z3QDS4n2rJZuWr/view?usp=sharing) | GRAPH,AUDIO,NONE |
| **G**-{ν,τ} | R34 | 61.2 | 85.7 | 64.8 | [`link`](https://drive.google.com/file/d/1cupR1HazNukPpSwFteYAtrmUf8ZiHkt-/view?usp=sharing) | GRAPH,TEXT,NONE |
| **G**-{ν,α,τ} | R34 | 61.5 | 85.7 | 65.6 | [`link`](https://drive.google.com/file/d/14ZoijXC9vichFMyJIiLYX1PHs5liqGT-/view?usp=sharing) | GRAPH,AUDIO_TEXT,SUM |
| **T**-{ν,α} | R34 | 61.8 | 85.8 | 66.0 | [`link`](https://drive.google.com/file/d/1EG6bzWbCG7cXCYw-ZurvSTMECLmivR0v/view?usp=sharing) | TRSFMR,AUDIO,NONE |
| **T**-{ν,τ} | R34 | 61.7 | 86.2 | 66.1 | [`link`](https://drive.google.com/file/d/1gXP_eBz_9jkXJJ5LZlvDtdXzqFn08rdo/view?usp=sharing) | TRSFMR,TEXT,NONE |
| **T**-{ν,α,τ} | R34 | 61.6 | 86.2 | 66.4 | [`link`](https://drive.google.com/file/d/1OVdi_1EJ-jX88vTyLstDzRoBxXqgKzXV/view?usp=sharing) | TRSFMR,AUDIOTEXT,SUM |

Notes:
- The "MMIT_VERSION" in [scripts/run_mmit_r2plus1d.sh](scripts/run_mmit_r2plus1d.sh) needs to be set to "v2".
- Add "MODEL.NUM_CLASSES 292" to [scripts/run_mmit_r2plus1d.sh](scripts/run_mmit_r2plus1d.sh).
