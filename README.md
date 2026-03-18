# Predicting Neuromodulation Outcome for Parkinson’s Disease with Generative Virtual Brain Model

A preliminary code implementation for the paper "Predicting Neuromodulation Outcome for Parkinson’s Disease with Generative Virtual Brain Model".

## Overview

This repository contains the source code, pre-trained models, and data processing pipelines described in our work. Our proposed framework leverages generative modeling to bridge the gap between large-scale neuroimaging data and limited clinical samples, enabling precise prediction of neuromodulation outcomes.

#### 1. Framework Overview
Overview of the proposed framework: a generative virtual brain paradigm transfers dynamical priors from large-scale data to small clinical cohorts for individualized neuromodulation response prediction.

![Overview of the proposed framework](Figure/framework.png)

#### 2. Model Architecture
Architecture of the generative virtual brain model.
![Architecture of the generative virtual brain model](Figure/model_sturcture.png)

#### 3. Workflow & Counterfactual Analysis
Schematic illustration of the iVB-based workflow for predicting neuromodulation response and diagram depicting the calculation of the Counterfactual Brain Mismatch.

![iVB-based workflow and CBM calculation](Figure/CBM.png)

## Installation

```bash
git clone git@github.com:desomeboy/Foundation-Individual-Generative-Virtual-Brain.git
cd Foundation-Individual-Generative-Virtual-Brain

conda create -n vtb python=3.9 -y
conda activate vtb
pip install -r requirements.txt
```

If you plan to run the scripts in [`Data_process/`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/tree/main/Data_process), you will also need several external command-line tools:

- [`dcm2niix`](https://github.com/rordenlab/dcm2niix) for DICOM-to-NIfTI conversion
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) utilities such as `bet`, `fast`, `flirt`, `fnirt`, `slicetimer`, `mcflirt`, `fslmaths`, and `applywarp`

Please make sure these tools are installed and available in your `PATH` (and `FSLDIR` is correctly configured for FSL) before running the preprocessing pipeline.

## Data

Public datasets used in this work include ADNI, ABIDE, PPMI, HCP, and the AAL3 atlas.
### Public Data Sources
| Dataset | Content | Access |
|---------|---------|--------|
| **ADNI / ABIDE / PPMI** | Multimodal neuroimaging & clinical data | [IDA/LONI](https://ida.loni.usc.edu) |
| **HCP Young Adult** | 1,200 healthy subjects resting-state fMRI | [HCP Portal](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) |
| **AAL3 Atlas** | 166-ROI brain parcellation | [AAL3](https://www.gin.cnrs.fr/en/tools/aal/) |

> Access to ADNI/ABIDE/PPMI requires registration via IDA.

### Preprocessing & Labels
- **Preprocessing code**: All pipelines (motion correction, normalization, AAL3 parcellation) are implemented in [`Data_process/`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/tree/main/Data_process).
- **Subject labels**: Subject IDs and clinical outcomes are stored in [`Data_process/Data_label.csv`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/blob/main/Data_process/Data_label.csv).

Because of data share restrictions, dataset locations are not hard-coded for public release. Please update local paths in [`vtb/config.py`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/blob/main/vtb/config.py) and related scripts according to your own environment. The data processing part in [`Data_process/`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/tree/main/Data_process) is provided for reference.


## Running Pipeline

The recommended workflow is:

1. Pre-train the foundation virtual brain (FVB) with [`scripts/train_FVB.py`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/blob/main/scripts/train_FVB.py).
2. Build subject-specific VTBs from clinical data with [`batch_iVB_final.sh`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/blob/main/batch_iVB_final.sh), which batch-runs [`scripts/train_iVB.py`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/blob/main/scripts/train_iVB.py).
3. Run downstream prognosis experiments in [`predict_exp/`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/tree/main/predict_exp).

### 1. Pre-train FVB

Train the base model on large-scale datasets:

```bash
python scripts/train_FVB.py --train
```

This step produces the pre-trained foundation model checkpoint used in later individualized analysis.

### 2. Build VTB for Clinical Subjects

After preparing clinical CSV files and setting the input/output paths in [`batch_iVB_final.sh`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/blob/main/batch_iVB_final.sh), run:

```bash
bash batch_iVB_final.sh
```

This step fine-tunes the pre-trained model for each clinical subject and saves the corresponding VTB outputs, including subject-level anomaly/distortion features.

### 3. Prognosis Evaluation

Use the scripts under [`predict_exp/`](https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain/tree/main/predict_exp) for downstream prediction tasks. For example:

```bash
python predict_exp/DBS/diff/PP_diff.py
```

or

```bash
python predict_exp/DBS/diff_regression/PP_diff_regression.py
```

Please update the input VTB directory and clinical table path inside each script before running.
