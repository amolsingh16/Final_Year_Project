# Krishi Mitra – Updated Research Paper Figures & Evaluation Data

This directory contains publication-ready figures, high-resolution vector exports, and raw metric tables generated from the **latest verified evaluation** of Krishi Mitra on the independent PlantVillage Tomato held-out test split (3,203 images, 10 classes).

All metrics have been directly computed from actual model forward passes and have not been approximated, estimated, or artificially inflated.

---

## 1. Generated Figures Index

| Figure File | Description | Recommended Paper Caption |
|---|---|---|
| `system_architecture_flowchart.png` | End-to-end Krishi Mitra diagnostic system architecture and OpenCV computer vision pipeline. | **Figure 1.** End-to-end system architecture and computer vision diagnostic workflow of Krishi Mitra. |
| `fig1_model_accuracy_comparison.png` | Model accuracy bar chart comparing MobileNetV2, CNN Baseline, and KrishiMitra Proposed. | **Figure 2.** Accuracy comparison of the evaluated models on the PlantVillage tomato test dataset. |
| `fig2_model_precision_comparison.png` | Macro precision bar chart across evaluated architectures. | **Figure 3.** Precision comparison of the evaluated models. |
| `fig3_model_recall_comparison.png` | Macro recall bar chart across evaluated architectures. | **Figure 4.** Recall comparison of the evaluated models. |
| `fig4_model_f1_score_comparison.png` | Macro F1-score bar chart across evaluated architectures. | **Figure 5.** F1-score comparison of the evaluated models. |
| `fig5_categorical_crossentropy_loss_comparison.png` | Categorical cross-entropy loss comparison. | **Figure 6.** Loss comparison of the evaluated models. |
| `fig6_classwise_disease_f1_score.png` | Per-class diagnostic F1-score line plot for 10 tomato categories. | **Figure 7.** Class-wise diagnostic F1-score performance across 10 tomato leaf disease categories. |
| `fig7_confusion_matrix_krishimitra.png` | Full normalized & count confusion matrix for the proposed ensemble. | **Figure 8.** Confusion matrix of the proposed KrishiMitra multi-model ensemble on 3,203 held-out test samples. |
| `fig8_per_class_precision_recall_f1.png` | Grouped bar chart comparing Precision, Recall, and F1 per class. | **Figure 9.** Per-class precision, recall, and F1-score breakdown for the proposed KrishiMitra model. |
| `fig11_dataset_class_distribution.png` | Dataset class distribution (Train: 12,794, Test: 3,203). | **Figure 10.** PlantVillage tomato dataset distribution across 10 classes (80% train, 20% test). |
| `fig12_final_model_performance_summary.png` | Multi-metric benchmark comparing Accuracy, Precision, Recall, and F1. | **Figure 11.** Multi-metric performance summary benchmarking MobileNetV2, CNN Baseline, and KrishiMitra Proposed. |

*Note: All figures are provided in high-resolution PNG (300 DPI), SVG (vector), and PDF (vector) formats.*

---

## 2. Note on Training vs. Validation Curves (Items 9 & 10)
- **Status:** **Not Generated (Omitted to preserve academic integrity)**
- **Reason:** Epoch-by-epoch training logs (`history.history` loss/accuracy arrays across epochs) were not serialized or stored during the model training sessions in the repository. Generating artificial training curves without genuine per-epoch history would violate IEEE research ethics. The final evaluated test set weights are evaluated directly instead.

---

## 3. Data Tables Index
- `model_metrics_comparison.csv`: Comprehensive summary table containing Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted), and Cross-Entropy Loss for all evaluated models.
- `classification_report_krishimitra.csv`: Class-wise Precision, Recall, F1, and Support for the proposed KrishiMitra ensemble.
- `classification_report_mobilenetv2.csv`: Class-wise report for the standalone MobileNetV2 candidate.
- `classification_report_cnn_baseline.csv`: Class-wise report for the baseline 3-block CNN.
- `confusion_matrix_krishimitra.csv`: 10x10 raw count confusion matrix table.
- `evaluation_summary.txt`: Quick reference text dossier comparing old screenshot values with newly verified figures.
