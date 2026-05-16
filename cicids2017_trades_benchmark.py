
"""
CICIDS2017 Binary Benchmark for TRADES-Based NIDS.

Required files:
- data/Monday-WorkingHours.pcap_ISCX.csv
- data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
"""

import os
import time
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import TensorFlowV2Classifier

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
SAMPLE_SIZE = 120000
BATCH_SIZE = 512
EPOCHS_BASELINE = 10
EPOCHS_TRADES = 5
ATTACK_EPSILON = 0.1
PGD_EVAL_SIZE = 1000
FGSM_EVAL_SIZE = 1000
CW_EVAL_SIZE = 200

MONDAY_FILE = "data/Monday-WorkingHours.pcap_ISCX.csv"
FRIDAY_DDOS_FILE = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

def set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(SEED)

def create_mlp(input_dim, num_classes=2):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"), Dropout(0.4),
        Dense(128, activation="relu"), Dropout(0.3),
        Dense(64, activation="relu"), Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

def load_cicids2017():
    print("Loading CICIDS2017 files...")
    if not os.path.exists(MONDAY_FILE):
        raise FileNotFoundError(f"Missing file: {MONDAY_FILE}")
    if not os.path.exists(FRIDAY_DDOS_FILE):
        raise FileNotFoundError(f"Missing file: {FRIDAY_DDOS_FILE}")

    df_monday = clean_columns(pd.read_csv(MONDAY_FILE, low_memory=False))
    df_friday = clean_columns(pd.read_csv(FRIDAY_DDOS_FILE, low_memory=False))

    df = pd.concat([df_monday, df_friday], axis=0, ignore_index=True)
    df = clean_columns(df)

    if "Label" not in df.columns:
        raise ValueError("Could not find Label column. Check the CICIDS2017 CSV format.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df["binary_label"] = df["Label"].astype(str).str.strip().apply(
        lambda x: 0 if x.upper() == "BENIGN" else 1
    )

    X = df.drop(columns=["Label", "binary_label"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df["binary_label"].astype(int).values

    if SAMPLE_SIZE and len(X) > SAMPLE_SIZE:
        X, _, y, _ = train_test_split(X, y, train_size=SAMPLE_SIZE, stratify=y, random_state=SEED)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    print(f"Samples used: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Classes: BENIGN={np.sum(y==0)}, ATTACK={np.sum(y==1)}")
    return X_train, X_test, y_train, y_test, X_train.shape[1]

def evaluate_attacks(model, X_test, y_test, input_dim, num_classes=2):
    loss_fn = SparseCategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(
        model=model, nb_classes=num_classes, input_shape=(input_dim,),
        loss_object=loss_fn, clip_values=(0, 1)
    )

    clean_acc = accuracy_score(y_test, np.argmax(model.predict(X_test, verbose=0), axis=1))

    def eval_attack(attack_name, size):
        idx = np.random.choice(len(X_test), min(size, len(X_test)), replace=False)
        X_sub, y_sub = X_test[idx], y_test[idx]

        if attack_name == "PGD":
            attacker = ProjectedGradientDescent(estimator=classifier, eps=ATTACK_EPSILON, max_iter=10, verbose=False)
        elif attack_name == "FGSM":
            attacker = FastGradientMethod(classifier, eps=ATTACK_EPSILON)
        else:
            attacker = CarliniL2Method(classifier, confidence=0.0, max_iter=10, batch_size=32, verbose=False)

        x_adv = attacker.generate(x=X_sub)
        pred = np.argmax(model.predict(x_adv, verbose=0), axis=1)
        return accuracy_score(y_sub, pred)

    return {
        "Clean Accuracy": clean_acc,
        "PGD Robust Accuracy": eval_attack("PGD", PGD_EVAL_SIZE),
        "FGSM Robust Accuracy": eval_attack("FGSM", FGSM_EVAL_SIZE),
        "C&W Robust Accuracy": eval_attack("CW", CW_EVAL_SIZE),
    }

def train_trades(X_train, y_train, input_dim):
    model = create_mlp(input_dim, 2)
    loss_c = SparseCategoricalCrossentropy(from_logits=False)
    loss_r = KLDivergence()
    optimizer = Adam(0.001)
    beta = 1.0

    clf = TensorFlowV2Classifier(
        model=model, nb_classes=2, input_shape=(input_dim,),
        loss_object=loss_c, clip_values=(0, 1)
    )
    pgd = ProjectedGradientDescent(estimator=clf, eps=ATTACK_EPSILON, max_iter=10, verbose=False)

    ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(BATCH_SIZE)

    print("Training TRADES model...")
    for epoch in range(EPOCHS_TRADES):
        print(f"  TRADES epoch {epoch+1}/{EPOCHS_TRADES}")
        for x_batch, y_batch in ds:
            x_adv = pgd.generate(x=x_batch.numpy())
            with tf.GradientTape() as tape:
                pred_clean = model(x_batch, training=True)
                pred_adv = model(x_adv, training=True)
                l_clean = loss_c(y_batch, pred_clean)
                l_robust = loss_r(pred_clean, pred_adv)
                loss = l_clean + beta * l_robust
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return model

def deployment_metrics(model, X_test):
    os.makedirs("results", exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = "results/trades_model_CICIDS2017.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    dummy = X_test[:1].astype(np.float32)
    for _ in range(50):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    n = 1000
    start = time.perf_counter()
    for _ in range(n):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
    end = time.perf_counter()

    latency_ms = ((end - start) * 1000) / n
    throughput = 1000 / latency_ms

    return {
        "TFLite Size KB": size_kb,
        "Latency ms/packet": latency_ms,
        "Throughput packets/sec": throughput
    }

def main():
    os.makedirs("results", exist_ok=True)
    X_train, X_test, y_train, y_test, input_dim = load_cicids2017()

    print("Training baseline model...")
    baseline = create_mlp(input_dim, 2)
    baseline.fit(
        X_train, y_train,
        epochs=EPOCHS_BASELINE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    )

    trades = train_trades(X_train, y_train, input_dim)

    print("Evaluating baseline...")
    baseline_results = evaluate_attacks(baseline, X_test, y_test, input_dim)
    baseline_results["Model"] = "Baseline MLP"

    print("Evaluating TRADES...")
    trades_results = evaluate_attacks(trades, X_test, y_test, input_dim)
    trades_results["Model"] = "TRADES MLP"

    attack_df = pd.DataFrame([baseline_results, trades_results])
    attack_df = attack_df[["Model", "Clean Accuracy", "PGD Robust Accuracy", "FGSM Robust Accuracy", "C&W Robust Accuracy"]]
    attack_df.to_csv("results/attack_results_CICIDS2017.csv", index=False)

    deploy = deployment_metrics(trades, X_test)
    deploy_df = pd.DataFrame([{"Dataset": "CICIDS2017", "Model": "TRADES MLP", **deploy}])
    deploy_df.to_csv("results/deployment_metrics_CICIDS2017.csv", index=False)

    table_row = {
        "Dataset": "CICIDS2017",
        "Samples Used": f"Stratified sample of {len(X_train)+len(X_test)} flows from Monday BENIGN and Friday DDoS CSV files; train={len(X_train)}, test={len(X_test)}",
        "No. of Classes": 2,
        "Clean Accuracy": trades_results["Clean Accuracy"],
        "PGD Robust Accuracy": trades_results["PGD Robust Accuracy"],
        "FGSM Robust Accuracy": trades_results["FGSM Robust Accuracy"],
        "C&W Robust Accuracy": trades_results["C&W Robust Accuracy"],
        "Notes": "Binary BENIGN-vs-DDoS benchmark using CICIDS2017 MachineLearningCSV files."
    }
    pd.DataFrame([table_row]).to_csv("results/table10_cicids2017_results.csv", index=False)

    print("\nSaved outputs:")
    print("- results/attack_results_CICIDS2017.csv")
    print("- results/deployment_metrics_CICIDS2017.csv")
    print("- results/table10_cicids2017_results.csv")
    print("- results/trades_model_CICIDS2017.tflite")

    print("\nAttack results:")
    print(attack_df.to_string(index=False))
    print("\nDeployment:")
    print(deploy_df.to_string(index=False))
    print("\nTable row:")
    print(pd.DataFrame([table_row]).to_string(index=False))

if __name__ == "__main__":
    main()
