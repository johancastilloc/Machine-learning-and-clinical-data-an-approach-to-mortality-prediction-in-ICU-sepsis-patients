# Machine learning and clinical data: an approach to mortality prediction in ICU sepsis patients

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
│   └── Imputation.py
└── preprocessing/
    ├── Difference.ipynb
    ├── Preprocessing_eICU.ipynb
    ├── Preprocessing_mimic.ipynb
    └── Unite_flag.ipynb
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
  - `Imputation.py`: Script for data imputation using Miceforest.
- **preprocessing/**: Contains notebooks for data preprocessing.
  - `Difference.ipynb`: Notebook to identify differences between columns of different datasets.
  - `Preprocessing_eICU.ipynb`: Notebook for preprocessing data from the eICU dataset.
  - `Preprocessing_mimic.ipynb`: Notebook for preprocessing data from the MIMIC-IV dataset.
  - `Unite_flag.ipynb`: Notebook to merge data and add the hospital expiration flag column.

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
  - `preprocessing/Preprocessing_eICU.ipynb`
  - `preprocessing/Preprocessing_mimic.ipynb`
  - `preprocessing/Unite_flag.ipynb`
  - `preprocessing/Difference.ipynb`

- Data Imputation:
  - `Miceforest/Imputation.py`

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
