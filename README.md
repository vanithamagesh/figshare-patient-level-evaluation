# Image-level versus patient-level evaluation on the figshare brain tumor collection

Code, prediction files and figures for the paper

> **How Much of the 99% Is Patient Overlap? Image-Level versus Patient-Level Evaluation of Deep Brain Tumor Classifiers on the Figshare MRI Collection**
> V. Vanitha, M. Srivani, G. Arulkumaran, B. Bhasker, Oana Geman, Alexandru Burlacu (manuscript submitted).

Four classifiers (ResNet50, EfficientNet-B0, DenseNet201 and BiFAM-Next) were trained with the same protocol on two partitions of the three-class figshare collection: an image-level partition and a patient-level partition.

## Main findings (all from the files in `results/`)

- **The image-level partition shares patients between training and test.** 195 of the 197 test patients also have slices in the training set. For 83.5% of the test slices, the most similar training slice comes from the same patient (median correlation 0.924, against 0.825 in the patient-level partition).
- **Every model drops under patient-level testing,** by 1.8–2.2 points of accuracy and 2.2–2.4 points of macro F1:

  | Model | Image level: accuracy / macro F1 | Patient level: accuracy / macro F1 | Correct patients (of 46) |
  |---|---|---|---|
  | ResNet50 | 96.25 / 95.69 | 94.10 / 93.40 | 42 |
  | EfficientNet-B0 | 97.88 / 97.59 | 96.07 / 95.34 | 43 |
  | DenseNet201 | 98.37 / 98.12 | 96.56 / 95.94 | 45 |
  | BiFAM-Next | 98.37 / 98.22 | 96.39 / 95.83 | 45 |

- **Patient-level results are uncertain.** Patient-bootstrap intervals of the macro F1 were 8–12 points wide, and no paired difference to BiFAM-Next excluded zero.
- **A few patients cause most errors.** Three test patients held 77 of the 103 misclassified slices of the four models.

## Repository layout

```
src/            bifamnet.py, jobs_core.py, run_local.py (training code shared with BiFAM-Next)
colab_cells/    baselines_cell.py (trains ResNet50, EfficientNet-B0 and DenseNet201 on both partitions)
                leakage_and_hard_patients_cell.py, attention_cell.py, gallery_cell.py (analyses of the saved runs)
                retrain_baselines_with_checkpoints_cell.py (optional: keeps baseline checkpoints for Grad-CAM)
notebooks/      Paper2_figures.ipynb, Attention_gallery.ipynb (Colab versions of the cells)
analysis/       analysis.py, table_metrics.py, extra_analysis.py (statistics)
                figs.py, fig_errors.py, fig_attstats.py (figures)
results/        preds/ (slice predictions of the 4 models × 2 partitions), protocol_results.json, analysis.json,
                table_metrics.json, extra_analysis.json, leak_stats.json, attention_in_mask.csv
figures/        figures of the paper
```

`bifamlite` is the internal name of BiFAM-Next. It appears in the Colab cells.

## Reproducing the statistics and figures (no GPU needed)

From the repository root:

```bash
pip install -r requirements.txt
python analysis/analysis.py          # patient-level analysis -> results/analysis_recomputed.json (identical to analysis.json)
python analysis/table_metrics.py     # metrics with bootstrap intervals
python analysis/extra_analysis.py    # per-class results, calibration, error overlap
python analysis/figs.py              # protocol, gap, confusion, per-patient and concentration figures
python analysis/fig_errors.py        # error structure and confidence
python analysis/fig_attstats.py      # attention inside the tumor mask
```

**To retrain the models:**
1. Prepare the data and train BiFAM-Next with the notebook of the BiFAM-Next repository.
2. Run `colab_cells/baselines_cell.py` (jobs 17–19) to train the three standard networks.
3. Run `notebooks/Paper2_figures.ipynb` on the saved runs for the leakage, hard-patient and attention analyses.

## Data

The figshare collection (Cheng, doi:10.6084/m9.figshare.1512427) is not redistributed here. The data-preparation step writes the partition files with a fixed seed.

## Citation

If you use this code or the prediction files, please cite the paper (see `CITATION.cff`).
