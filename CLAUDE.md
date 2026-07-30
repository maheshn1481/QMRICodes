# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conventions (always follow)

- **Keep `README_qMRI_Workflow.md` current.** Whenever you add a script or change how an existing one is invoked (new/renamed flags, arguments, inputs/outputs, or workflow order), update `README_qMRI_Workflow.md` in the same change — add the new script to the workflow with an example command, and update the usage for changed flags. A code change that alters user-facing usage is not complete until the README reflects it.

## What this is

MATLAB research code for **physics-guided quantitative MRI (qMRI) parameter mapping**. A 3D U-Net takes three routine structural contrasts (T1w, T2w, FLAIR) as input and predicts six maps — PD, T1, T2, and three signal-gain terms g1/g2/g3. The key idea: predicted maps are pushed back through MR signal equations (SPGR/GRE, spin-echo, inversion-recovery) to re-synthesize the input images, and the loss compares synthesized-vs-measured signal plus an L1 term against reference SyMRI/MAGiC maps. There is no build system, package manager, or test suite — scripts are run interactively from within MATLAB.

## Running (no CLI, no tests)

Requires MATLAB with **Deep Learning Toolbox** + **Image Processing Toolbox** (Signal Processing Toolbox optional; GPU auto-detected via `canUseGPU`). Run scripts from within MATLAB with the current folder set to the data root — every script uses `config.rootDir = pwd` and expects `acq_params.xlsx` plus patient subfolders there. Prediction scripts pick model/Excel/root via `uigetfile`/`uigetdir` dialogs.

Recommended end-to-end workflow (see `README_qMRI_Workflow.md`):
1. Train (LOPO, auto): `run_qmri_3dcnn_NonNormalSignalLeaveOneOutTraining_Automatic.m` → writes `trained_models_LeftOut_<PID>/`
2. Predict: `predict_qmri_3dcnn_NonNormalSignal_AutomatedAnonymizedFolders.m` → writes `<PID>/<savedir>/*_pred.nii.gz`
3. Figures: `ManuscriptPlotsGenerationCode.m` (run on a server), then `ReopenfigfilesandSave.m` to re-export clean `.fig`/`.png`.

## Current MAGiC CSV cohort: DICOM→NIfTI preprocessing + 5-fold CV

The two **recommended** scripts above now operate in **CSV / 5-fold-CV mode** for the anonymized MAGiC cohort (schema in `doc/fulldataset.md`), not the legacy `acq_params.xlsx` + per-`PatientID`-folder mode described below. The cohort is indexed by `dataset.csv` (key column `AnonymizationID`, e.g. zero-padded `"000"`).

Because the CSV's contrasts/maps are **DICOM**, a Python/SimpleITK step runs **before** MATLAB — `preprocess_dicom_to_nifti.py`:
- Converts `T1W/T2W/FLAIR Synthetic` (one DICOM series each) → `processed/<AnonymizationID>/{T1W,T2W,FLAIR}.nii.gz`, stamping TR/TE/FA/TI from the DICOM into each NIfTI header `descrip` (MATLAB reads it back via `niftiinfo().Description`; `readAcqParams`).
- Converts `SYMAPS` (a dir of per-slice `SYMAPS_<NN>_{T1,T2,PD}.dcm`) → `T1map/T2map/PD.nii.gz` (the quantitative references; `RescaleSlope/Intercept` applied).
- `PS Synthetic` is intentionally **not** converted (phase-sensitive; unused).
- After converting, it checks **per-patient grid consistency by default** (`[grids] N/total need --resample`); `--check-grids` runs only that check on an existing dir. Analysis is **per-patient and masked-to-brain**, so only a *single patient's* SYMAPS-vs-weighted grid mismatch matters — inter-patient slice-count differences are irrelevant. `--resample` mirrors a patient onto the T1W grid in a **separate** `processed_resampled/` dir when needed (else copies).
- Optional **brain mask** (skull-strip): `skullstrip_hdbet.py --processed processed` runs HD-BET on T1W → `processed/<id>/mask.nii.gz` (what MATLAB's `config.fileMask` expects; else it falls back to nonzero-T1W). HD-BET is an optional dep (`pip install HD-BET`).
- Setup: `python3 -m venv /opt/qmricodes && /opt/qmricodes/bin/pip install -r requirements.txt`.

**PACS series inventory:** `query_series_descriptions.py` walks `dataset.csv`, issues one SERIES-level C-FIND per `Study UID` with dcmtk's `findscu` (subprocess; `-X` extraction read back with pydicom, verbose-text-dump fallback for older dcmtk), and writes `series_index.csv` (AnonymizationID + Study/Series UID + SeriesNumber/Description/Modality/ProtocolName/instance count) plus a UID-free `series_find_log.csv`. Its CLI deliberately mirrors `../dicomtransfer/batch_cmove.py` (same `--aet/--aec/--host/--port/--has-header/--study-col/--retries/--retry-wait/--timeout/--log/--dry-run`), adding `--id-col/--out/--findscu/--resume/--summary/--show-uids`. **dcmtk is a system package** (`apt install dcmtk`), not a pip dep. Console output prints only `AnonymizationID` + counts; `--show-uids` is the debug-only analogue of `--show-paths`. Both output CSVs are gitignored — `series_index.csv` contains UIDs. `--summary` prints the PHI-free description→patient-count histogram, which is how to check whether a real post-contrast T1 exists in the cohort.

**MIST tumor model (separate `/home/fuentes/github/MIST` repo):** its tumor net is a fixed **4-channel** `t1,t2,tc,fl` net with **no missing-modality support**. This cohort has **no TC** (post-contrast T1), which **cannot be synthesized from MAGiC**. To apply MIST you must either retrain a 3-channel (T1/T2/FLAIR) model (needs tumor labels), feed a T1-as-TC surrogate (degraded ET/tumor-core), or supply real T1c from PACS. Not yet implemented — decision pending.

MATLAB side (both recommended scripts): read `dataset.csv` (+ PHI-free `cv_folds.csv` manifest), group patients by `AnonymizationID` into 5 folds, models → `trained_models_Fold<f>/`, predictions → `<outRoot>/<AnonymizationID>/Fold<f>_Predictions/`. Set `config.processedRoot` (default `<rootDir>/processed`; point at `processed_resampled/` if you ran `--resample`). Acq params come from the NIfTI `descrip`, not the CSV.

**PHI — read this (`../radpathsandbox/CLAUDE.md` rules apply):** the `dataset.csv` path columns (`dicom_path`, `synthentic_path`, `*_Synthetic`, `SYMAPS`) can embed PHI in directory names. `preprocess_dicom_to_nifti.py` therefore **hides literal paths in its diagnostics by default**, printing only resolution status + `AnonymizationID`. The `--show-paths` flag reveals full paths and is for **local debugging only — never paste/commit its output**. Do not add diagnostics that echo these paths by default. See the `cohort-csv-paths-are-phi` project memory.

Inspect a converted patient in ITK-SNAP: `./view_patient.sh <AnonymizationID> [processed_dir]` (loads T1W as main, the rest as overlays; use `list` to enumerate patients; pass `processed_resampled` to view the resampled set).

## Architecture

**Script-oriented with heavy duplication, not a shared library.** Each runnable `.m` inlines its own copies of helpers (`readNii`, `pickVar`, `matchExisting`, `synthSignals`, `sigmoid`, `normalizeContrasts`, …). When fixing a helper, expect to change it in several files. The only genuinely shared function is `ccc_barnes.m` (concordance correlation coefficient, the primary agreement metric). Coupling between scripts is by **data flow on disk**, not function calls:

    training scripts → trained_models_*/{best_epoch.mat, epochNNN.mat, LossesFigure.fig}
      → prediction scripts consume the .mat (net + config) → <PID>/<savedir>/*_pred.nii.gz + Comparison.fig + TestLosses.mat
        → ManuscriptPlotsGenerationCode.m consumes prediction outputs → ReopenfigfilesandSave.m re-exports its .fig
    TrainingAndValidationLossDataExtractor.m consumes the training LossesFigure.fig files

The core model + training loop lives in local functions of the training scripts (`buildUNet3D`, `extractPatches3D`, `makeMinibatchQueue`, `modelGradients`, `synthSignals`, `adamw_step`, `cosineLR`). Prediction scripts re-implement these plus `slidingPredict` (sliding-window inference with Gaussian blending). Output ranges are hard-capped via scaled sigmoid (PD≤150, T1≤5000 ms, T2≤3000 ms, g≤500). The `config` struct (patch size, caps, hyperparameters) is saved inside every model `.mat` so prediction can reconstruct inference settings.

Script groups:
- **Training**: `run_qmri_3dcnn_NonNormalSignalAllPatientsTraining.m` (all patients → `trained_models_All/`), `...LeaveOneOutTraining.m` (manual LOPO, needs `acq_paramsP1.xlsx`, `acq_paramsP2.xlsx`, …), `...LeaveOneOutTraining_Automatic.m` (recommended; loops folds from one `acq_params.xlsx`).
- **Prediction**: `predict_qmri_3dcnn_NonNormalSignal.m` (manual), `..._Automated.m`, `..._AutomatedAnonymizedFolders.m` (recommended).
- **Post-processing**: `ManuscriptPlotsGenerationCode.m`, `TrainingAndValidationLossDataExtractor.m`, `ReopenfigfilesandSave.m`.
- **Data prep / utilities**: `DataProcessing.m`, `DataProcessingIndPatient.m`, `ConvertPatientDatatoCSV.m`, `dicomLoaderAndViewer.m` (standalone DICOM viewer), `ccc_barnes.m`.

## Data conventions

- `acq_params.xlsx` must have a `PatientID` column plus acquisition params `TRT1, TET1, FAT1_deg, TRT2, TET2, TRFLAIR, TEFLAIR, TIFLAIR`. Column names are alias-resolved via the repeated `pickVar` helper (`TR_T1`, `TR_Gre`, `TR_SPGR`, …).
- **Patient folder names must equal the `PatientID`.** Per-folder files: inputs `PREOP_T1_W_S.nii.gz`, `PREOP_AxT2_W_S.nii.gz`, `PREOP_FLAIR_W_S.nii.gz`; references `pdmap.nii.gz`, `t1map.nii.gz`, `t2map.nii.gz`; optional `mask.nii.gz` (skull-stripped brain). All I/O is NIfTI.
- Two data roots referenced in docs: `ProcessedData` (real Patient IDs) and `ProcessedData2` (anonymized `P0001…`).

## Gotchas

- Several data-prep scripts have **hard-coded Windows absolute paths** (e.g. `C:\Users\MNandyala\...` in `ConvertPatientDatatoCSV.m`, `DataProcessing.m`) that must be edited before running elsewhere.
- One readme filename contains a literal space: `readme_ run_qmri_3dcnn_NonNormalSingalAllPatientsTraining.txt`.
- `README_qMRI_Workflow.md` is the single workflow doc (the old `.txt` duplicate was removed).
- TV / edge-aware regularizers (`tvCharb`, `l2smooth`, `computeEdgeWeights`) are implemented but currently commented out in the loss.
- PHI: never print/commit data values, and treat `dataset.csv` path columns as PHI (see the "Current MAGiC CSV cohort" section and `../radpathsandbox/CLAUDE.md`). Work with column names, counts, and status only. `--show-paths` is debug-only.
