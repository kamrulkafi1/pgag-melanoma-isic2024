# Patient-Guided Attention Gate for False-Negative-Aware Melanoma Detection

Code and experimental notebook for the paper *"Patient-Guided Attention Gate for
False-Negative-Aware Melanoma Detection Using Hybrid CNN-ViT Architecture on
ISIC 2024 SLICE-3D"* (AII 2026).

## Overview

A hybrid EfficientNet-B4 + Vision Transformer (ViT-B/16) architecture with a
novel **Patient-Guided Attention Gate (PGAG)** that conditions CNN/ViT feature
fusion on a 7-dimensional patient-context deviation vector (a computational
operationalisation of the clinical *ugly duckling sign*), combined with an
asymmetric false-negative-aware focal loss for extreme class imbalance
(0.1% malignant prevalence).

**Key result:** pAUC = 0.1892 (FPR ≤ 0.20) on the ISIC 2024 SLICE-3D
validation set, exceeding the competition-winning submission (pAUC = 0.1755)
by 7.8%.

## Repository contents

- `notebook/` — the full experimental notebook (data loading, feature
  engineering, model, training, evaluation, ablation, figures)
- `README.md` — this file

## Large files (checkpoints, features)

Trained model checkpoints and pre-extracted ViT-B/16 feature files are hosted
separately on Zenodo due to file size (GitHub's 100 MB per-file limit):

**DOI: [INSERT ZENODO DOI HERE]**

Files provided there:
- `best_model_main_final.pt` — Phase 1 (main training) checkpoint
- `best_hybrid_model_gatesup_final.pt` — Phase 2 (gate fine-tuning) checkpoint,
  used for all reported results
- ViT-B/16 pre-extracted feature file (parquet)

## Requirements

```
torch>=2.0
timm
pandas
numpy
scikit-learn
shap
albumentations
matplotlib
seaborn
```

## Data

The ISIC 2024 SLICE-3D dataset is publicly available at
https://challenge.isic-archive.com/data/#2024.

## Reproducing results

1. Download the ISIC 2024 SLICE-3D dataset and pre-extracted ViT features
   (see above)
2. Open `notebook/` in Jupyter or Google Colab
3. Run cells in order; see in-notebook comments for cell-by-cell descriptions
4. To skip retraining and reproduce evaluation/figures only, load the
   Phase 2 checkpoint from Zenodo and start from the metrics cell

## Citation

If you use this code, please cite:

```
[INSERT FULL CITATION AFTER PROCEEDINGS PUBLICATION]
```

## License

[INSERT LICENSE — e.g. MIT, or match your institution's policy]

## Contact

For questions or data-access requests, contact: kamrul.kafi1@gmail.com
