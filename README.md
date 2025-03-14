# Multicenter validation of a machine learning algorithm for mortality prediction at decision-critical timestamps in ICU sepsis patients

This research aims to predict mortality for ICU sepsis patients at 12, 24, and 48 hours in advance through machine learning models based on clinical data. The project includes data preprocessing, missing value imputation, model fine-tuning, and model evaluation.

## Project Structure

```
├── environment.yml
├── Models_12h.ipynb
├── Models_24h.ipynb
├── Models_48h.ipynb
├── confidenceinterval/
│   ├── goldstandard.py
│   ├── ModelEvaluationUtils.py
│   └── __pycache__/
│       ├── goldstandard.cpython-312.pyc
│       └── ModelEvaluationUtils.cpython-312.pyc
├── Fine_tuning/
│   ├── Optuna_12h.ipynb
│   ├── Optuna_24h.ipynb
│   ├── Optuna_48h.ipynb
├── Miceforest/
│   └── imputation.py
└── preprocessing/
    ├── Diferencia.ipynb
    ├── preprocessing_eICU.ipynb
    ├── preprocessing_mimic.ipynb
    └── unir_flag.ipynb
```

### Files and Directories

- **environment.yml**: Configuration file to create the conda environment required to run the project.
- **Models_12h.ipynb, Models_24h.ipynb, Models_48h.ipynb**: Notebooks containing the implementation of models for predictions at 12, 24, and 48 hours respectively.
- **confidenceinterval/**: Contains scripts for model evaluation and confidence interval calculation.
  - `goldstandard.py`: Functions for model evaluation.
  - `ModelEvaluationUtils.py`: Utilities for model evaluation.
- **Fine_tuning/**: Contains notebooks for model fine-tuning using Optuna.
  - `Optuna_12h.ipynb, Optuna_24h.ipynb, Optuna_48h.ipynb`: Notebooks for model fine-tuning at 12, 24, and 48 hours respectively.
- **Miceforest/**: Contains scripts for missing value imputation.
  - `imputation.py`: Script for data imputation using Miceforest.
- **preprocessing/**: Contains notebooks for data preprocessing.
  - `Diferencia.ipynb`: Notebook to identify differences between columns of different datasets.
  - `preprocessing_eICU.ipynb`: Notebook for preprocessing data from the eICU dataset.
  - `preprocessing_mimic.ipynb`: Notebook for preprocessing data from the MIMIC-IV dataset.
  - `unir_flag.ipynb`: Notebook to merge data and add the hospital expiration flag column.

## Installation

To install the necessary dependencies, run the following command:

```sh
conda env create -f environment.yml
```

## Usage

1. Activate the conda environment:

```sh
conda activate sepsis
```

2. Run the notebooks in the following order to reproduce the results:

- Data Preprocessing:
  - `preprocessing/preprocessing_eICU.ipynb`
  - `preprocessing/preprocessing_mimic.ipynb`
  - `preprocessing/unir_flag.ipynb`
  - `preprocessing/Diferencia.ipynb`

- Data Imputation:
  - `Miceforest/imputation.py`

- Model Fine-Tuning:
  - `Fine_tuning/Optuna_12h.ipynb`
  - `Fine_tuning/Optuna_24h.ipynb`
  - `Fine_tuning/Optuna_48h.ipynb`

- Model Evaluation:
  - `Models_12h.ipynb`
  - `Models_24h.ipynb`
  - `Models_48h.ipynb`


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
