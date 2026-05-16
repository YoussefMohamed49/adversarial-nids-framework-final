CICIDS2017 Benchmark Instructions

Required folder structure:

adversarial-nids-framework-final/
├── cicids2017_trades_benchmark.py
├── requirements_updated.txt
├── data/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

Run from the main project folder:

py -3.11 -m pip install -r requirements_updated.txt
py -3.11 cicids2017_trades_benchmark.py

Expected outputs:

results/table10_cicids2017_results.csv
results/attack_results_CICIDS2017.csv
results/deployment_metrics_CICIDS2017.csv

If runtime is slow, open cicids2017_trades_benchmark.py and reduce:

SAMPLE_SIZE = 120000

to:

SAMPLE_SIZE = 50000
