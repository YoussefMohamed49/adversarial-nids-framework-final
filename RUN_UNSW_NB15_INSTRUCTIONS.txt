UNSW-NB15 Benchmark Instructions

Required folder structure:

adversarial-nids-framework-final/
├── main_with_unsw_nb15.py
├── requirements_updated.txt
├── data/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv

Run from the main project folder:

py -3.11 -m pip install -r requirements_updated.txt
py -3.11 main_with_unsw_nb15.py

Expected outputs:

results/table10_cross_dataset_results.csv
results/attack_results_UNSW-NB15.csv
results/deployment_metrics_UNSW-NB15.csv
