# Adversarial Defense Framework for Network Intrusion Detection

This repository contains a Python framework for evaluating the robustness of Neural Network-based Network Intrusion Detection Systems (NIDS) against adversarial attacks. It implements a baseline model and a defensively trained model using TRADES, and evaluates them on the NSL-KDD and Bot-IoT datasets.

## Features
- **Two Datasets**: Supports both **NSL-KDD** and **Bot-IoT**.
- **Baseline Model**: A standard Multi-Layer Perceptron (MLP) for classification.
- **Defended Model**: An MLP trained with the **TRADES** (TRadeoff-inspired Adversarial DEfense via Surrogate loss) method for improved robustness.
- **Systematic Evaluation**: Compares model performance on both clean and adversarial data (PGD attacks).
- **Visualizations**: Generates plots for accuracy comparison, per-class F1-scores, and the accuracy-robustness trade-off.
- **IoT Deployment**: Prepares and converts the final defended model to a TensorFlow Lite (`.tflite`) format for resource-constrained devices.

## Project Structure
adversarial-nids-framework/
├── data/               # Place datasets (e.g., KDDTrain+.txt, Bot-IoT.csv) here
├── results/            # Git-ignored folder where output plots are saved
├── .gitignore          # Specifies files for Git to ignore
├── main.py             # The main executable script
├── README.md           # Project documentation
└── requirements.txt    # Required Python libraries


## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd adversarial-nids-framework
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add your datasets** to the `data/` folder.

## How to Run

The main script `main.py` can be configured to run with either the NSL-KDD or Bot-IoT dataset.

-   Open `main.py` and find the line: `USE_BOT_IOT = False`
-   Set it to `False` to use the **NSL-KDD** dataset.
-   Set it to `True` to use the **Bot-IoT** dataset.

Then, run the script from your terminal:
```bash
python main.py