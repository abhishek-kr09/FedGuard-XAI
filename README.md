# FedGuard-XAI

FedGuard-XAI is an explainable federated intrusion-detection system for studying poisoned clients, statistical update detection, robust aggregation, and explanation drift. The project uses CIC-IDS2017 as its primary dataset and keeps dataset files and generated results outside version control.

## Problem statement

Federated learning allows network participants to train an intrusion-detection model without centralizing raw traffic. A malicious participant can still manipulate local labels or model updates and degrade the shared model. FedGuard-XAI studies this threat and combines three signals:

- statistical client-update anomalies such as L2 norm and cosine similarity;
- model explanations generated with SHAP;
- explanation drift measured against a reference explanation using Jensen-Shannon divergence.

The final system trains clients, detects suspicious behavior, filters or robustly aggregates updates, evaluates the global model, and exposes the results through a Streamlit dashboard.

## Phase roadmap

### ✅ Completed - Phase 0: Project setup

Set up the Git repository, Python 3.12 environment, pinned dependencies, directory layout, and `.gitignore`. Verify that the environment and test discovery are reproducible.

### ✅ Completed - Phase 1: Dataset analysis

Loaded all eight CIC-IDS2017 CSV files and inspected columns, labels, class distribution, missing values, infinity values, chunk-level duplicates, and basic traffic statistics. Produced exploratory tables and class-distribution visualizations in `notebooks/01_data_analysis.ipynb`.

### ✅ Completed - Phase 2: Data preprocessing

Cleaned the CIC-IDS2017 data with streaming reads, replaced infinity values, mapped `BENIGN` to `0` and attacks to `1`, removed per-file duplicates, created reproducible train/validation/test splits, fit imputation and scaling on training data only, and saved the processed arrays and preprocessor from `notebooks/02_data_preprocessing.ipynb`.

### ✅ Completed - Phase 3: Baseline IDS

Built and evaluated a centralized PyTorch MLP using the Phase 2 processed arrays. The baseline trained on 200,000 rows and evaluated on 50,000 validation rows:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9344 |
| Precision | 0.9762 |
| Recall | 0.8706 |
| F1-score | 0.9204 |

The confusion matrix in `notebooks/03_baseline_ids.ipynb` was `[[27741, 462], [2820, 18977]]`, using `[[TN, FP], [FN, TP]]` ordering. The full test suite passed with 7 tests.

### Phase 4 - Federated learning

Partition processed data across multiple clients and implement local PyTorch training with Flower. Begin with honest clients and FedAvg over multiple communication rounds.

### Phase 5 - Poisoning attack

Introduce a malicious client that performs label flipping. Compare clean and poisoned federated learning and quantify the effect on global performance.

### Phase 6 - Poisoned-client detection

Analyze client updates with L2 norms, cosine similarity, and related statistical signals. Produce anomaly scores and identify potentially poisoned clients.

### Phase 7 - Defense and robust aggregation

Filter, exclude, or robustly aggregate suspicious updates. Compare FedAvg with defended aggregation using global accuracy, F1, detection rate, and false-positive rate.

### Phase 8 - SHAP explainability

Apply SHAP to the global model and individual client models to produce feature-attribution vectors and visualizations for IDS predictions.

### Phase 9 - Explanation drift

Compare client SHAP explanations with a reference explanation. Calculate explanation drift with Jensen-Shannon divergence and compare clean versus poisoned clients.

### Phase 10 - Complete FedGuard-XAI pipeline

Combine statistical detection, round-based analysis, SHAP explanation drift, and client filtering into one client-training, detection, defense, and global-model pipeline.

### Phase 11 - Experiments and evaluation

Run clean FL, poisoned FL, statistical detection, SHAP, explanation-drift, and combined-defense experiments. Repeat controlled seeds and configurations and record metrics, plots, means, and standard deviations.

### Phase 12 - Dashboard

Build a Streamlit dashboard for global performance, federated rounds, client status, anomaly scores, and SHAP explanations. The dashboard reads saved experiment results and does not contain model-training logic.

### Phase 13 - Documentation

Document the architecture, dataset instructions, methodology, setup, experiments, results, reproducibility steps, and role of every module.

### Phase 14 - Final demo and viva

Demonstrate the complete flow from dataset to IDS, FL, poisoning, detection, SHAP drift, defense, and final model. Prepare the report, presentation, architecture diagrams, experimental results, and technical/viva answers.

## Phase 0 setup

FedGuard-XAI targets **CPython 3.12**. The supported version is constrained in `pyproject.toml` and reflected in the pinned `requirements.txt`.

### Windows PowerShell

Use Python 3.12 to create and activate the project environment:

```powershell
cd D:\FedGuard-XAI
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The expected interpreter output begins with `Python 3.12`. The editable install makes the local `src` package importable while `--no-deps` avoids installing a second dependency set.

### Repository initialization

From the project root:

```powershell
git init
git add .
git commit -m "Initialize FedGuard-XAI project"
git branch -M main
```

Create an empty GitHub repository named `FedGuard-XAI`, then connect it with the repository URL supplied by GitHub:

```powershell
git remote add origin <GITHUB_REPOSITORY_URL>
git push -u origin main
```

Do not commit CIC-IDS2017 files, virtual environments, secrets, caches, or generated experiment outputs. The current `.gitignore` preserves only the required directory placeholders.

### Phase 0 verification

```powershell
python --version
python -c "import flwr, numpy, pandas, sklearn, torch; print('dependencies ok')"
python -m pytest
python -m compileall src experiments dashboard
```

Phase 0 is complete when the version check reports Python 3.12, the dependency import check succeeds, tests are collected, and compilation completes without errors. Later phases will add the PyTorch MLP and Flower simulation on top of this foundation.

## Dataset

Download CIC-IDS2017 from its official source and place the extracted CSV files under `data/raw/`. Do not commit the dataset. The expected analysis workflow will discover CSV files recursively, normalize column names, and record the source files used for each experiment. Processed, model-ready data belongs under `data/processed/`.

## Current project structure

```text
data/                 Raw and processed datasets
src/                  Reusable preprocessing, models, FL, defense, and XAI code
experiments/configs/  Reproducible YAML configurations
experiments/scripts/  Experiment entry points
results/              Metrics, plots, and logs generated by runs
notebooks/            Analysis checkpoints for Phases 1, 3, 4, and 8
dashboard/            Streamlit result viewer
tests/                Unit and regression tests
```

## Existing quick checks

```powershell
python experiments\scripts\run_clean_fl.py
python experiments\scripts\run_poisoned_fl.py
python experiments\scripts\run_defense.py
python experiments\scripts\run_xai.py
streamlit run dashboard\app.py
```

The current scripts are small smoke-test entry points. The phase implementation will expand them into dataset-backed, round-based experiments while keeping result generation separate from the dashboard.

## Module responsibilities

- `src/preprocessing.py`: CSV loading, missing-value handling, scaling, and categorical encoding.
- `src/model.py`: PyTorch MLP, legacy Random Forest baseline, and evaluation metrics.
- `src/client.py` and `src/server.py`: client updates and federated orchestration.
- `src/poisoning.py`: reproducible binary label-flip attack.
- `src/detector.py`: robust aggregation and update anomaly detection.
- `src/xai.py`: model-native feature importance and lazy SHAP integration.
- `notebooks/`: exploratory and explanatory checkpoints.
- `dashboard/app.py`: result visualization only; training belongs in experiment scripts.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
