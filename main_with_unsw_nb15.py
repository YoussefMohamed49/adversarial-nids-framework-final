"""
Hardening the IoT Edge: TRADES-based NIDS Robustness Experiment
Updated version: adds UNSW-NB15 binary benchmarking and reviewer-ready result export.

Supported datasets:
- NSL-KDD
- Bot-IoT
- UNSW-NB15

Outputs saved in ./results/:
- table10_cross_dataset_results.csv
- attack_results_<dataset>.csv
- deployment_metrics_<dataset>.csv
- per_class_report_<dataset>_<model>.csv
- trades_model_<dataset>.tflite
- generated figures
"""

import os
import time
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import TensorFlowV2Classifier

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
# 1. Configuration
# ============================================================

DATASET = "UNSW_NB15"  # Options: "NSL_KDD", "BOT_IOT", "UNSW_NB15"

NUM_RUNS = 1
EPOCHS_BASELINE = 15
EPOCHS_ADV = 5
BATCH_SIZE = 512
ATTACK_EPSILON = 0.1
BETA = 1.0
RANDOM_SEED = 42

# Reviewer-friendly evaluation sizes. Increase if hardware allows.
PGD_EVAL_SIZE = 1000
FGSM_EVAL_SIZE = 1000
CW_EVAL_SIZE = 200

# Relative paths for reproducibility
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NSL_KDD_DIR = os.path.join(DATA_DIR, "NSL-KDD")
BOT_IOT_FILE = os.path.join(DATA_DIR, "reduced_data_4.csv")
UNSW_TRAIN_FILE = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
UNSW_TEST_FILE = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Optional: force CPU for comparable edge/deployment measurement
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def set_seeds(seed_value=42):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


set_seeds(RANDOM_SEED)

# ============================================================
# 2. Dataset Loaders
# ============================================================

def load_and_preprocess_unsw_nb15(train_file, test_file, binary=True, apply_smote=False):
    """Loads official UNSW-NB15 training/testing CSV files.

    binary=True uses label column: 0 = Normal, 1 = Attack.
    binary=False uses attack_cat for multiclass classification.
    """
    print(f"Loading UNSW-NB15 training file: {train_file}")
    print(f"Loading UNSW-NB15 testing file:  {test_file}")

    df_train = pd.read_csv(train_file, low_memory=False)
    df_test = pd.read_csv(test_file, low_memory=False)

    # Basic cleanup
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    # Target
    if binary:
        y_train_raw = df_train["label"].astype(int)
        y_test_raw = df_test["label"].astype(int)
        class_names = np.array(["Normal", "Attack"])
    else:
        y_train_raw = df_train["attack_cat"].fillna("Normal").astype(str).str.strip()
        y_test_raw = df_test["attack_cat"].fillna("Normal").astype(str).str.strip()
        le = LabelEncoder()
        y_train_raw = le.fit_transform(y_train_raw)
        y_test_raw = le.transform(y_test_raw)
        class_names = le.classes_

    # Drop ID and target columns. Keep only features.
    drop_cols = ["id", "attack_cat", "label"]
    X_train = df_train.drop(columns=drop_cols, errors="ignore")
    X_test = df_test.drop(columns=drop_cols, errors="ignore")

    # Replace infinities and missing values
    X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    X_full = X_full.replace([np.inf, -np.inf], np.nan)

    categorical_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X_full.columns if c not in categorical_cols]

    for c in numeric_cols:
        X_full[c] = pd.to_numeric(X_full[c], errors="coerce")
        X_full[c] = X_full[c].fillna(X_full[c].median())

    for c in categorical_cols:
        X_full[c] = X_full[c].fillna("unknown").astype(str)

    X_full = pd.get_dummies(X_full, columns=categorical_cols)

    X_train_enc = X_full.iloc[:len(df_train)].copy()
    X_test_enc = X_full.iloc[len(df_train):].copy()

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    y_train = np.array(y_train_raw)
    y_test = np.array(y_test_raw)

    if apply_smote:
        print(f"  [Pre-SMOTE] Training size: {X_train_scaled.shape[0]}")
        smote = SMOTE(random_state=RANDOM_SEED)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"  [Post-SMOTE] Training size: {X_train_scaled.shape[0]}")

    return (
        (X_train_scaled.astype(np.float32), y_train),
        (X_test_scaled.astype(np.float32), y_test),
        class_names,
        X_train_scaled.shape[1],
        {"train_size": len(df_train), "test_size": len(df_test), "binary": binary}
    )


def load_and_preprocess_nsl_kdd(data_path):
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    train_path = os.path.join(data_path, 'KDDTrain+.txt')
    test_path = os.path.join(data_path, 'KDDTest+.txt')
    df_train = pd.read_csv(train_path, header=None, names=columns)
    df_test = pd.read_csv(test_path, header=None, names=columns)
    df_train.drop('difficulty', axis=1, inplace=True)
    df_test.drop('difficulty', axis=1, inplace=True)

    label_mapping = {
        'normal': 'normal',
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos',
        'mailbomb': 'dos', 'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l',
        'warezclient': 'r2l', 'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l',
        'snmpguess': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r', 'httptunnel': 'u2r',
        'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
    }
    df_train['label'] = df_train['label'].map(label_mapping)
    df_test['label'] = df_test['label'].map(label_mapping)
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    categorical_cols = ['protocol_type', 'service', 'flag']
    df_full = pd.concat([df_train, df_test], axis=0)
    df_full_encoded = pd.get_dummies(df_full, columns=categorical_cols)
    df_train_encoded = df_full_encoded.iloc[:len(df_train)]
    df_test_encoded = df_full_encoded.iloc[len(df_train):]

    le = LabelEncoder()
    y_train = le.fit_transform(df_train_encoded['label'])
    y_test = le.transform(df_test_encoded['label'])
    X_train = df_train_encoded.drop('label', axis=1)
    X_test = df_test_encoded.drop('label', axis=1)

    smote = SMOTE(random_state=RANDOM_SEED)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train.astype(np.float32), y_train), (X_test.astype(np.float32), y_test), le.classes_, X_train.shape[1], {"train_size": len(X_train), "test_size": len(X_test)}


def load_and_preprocess_bot_iot(file_path, sample_size=200000):
    df_full = pd.read_csv(file_path, low_memory=False)
    if sample_size > len(df_full):
        sample_size = len(df_full)
    df, _ = train_test_split(df_full, train_size=sample_size, stratify=df_full['category'], random_state=RANDOM_SEED)

    cols_to_drop = ['pkSeqID', 'stime', 'flgs', 'saddr', 'daddr', 'sport', 'dport', 'attack', 'category', 'subcategory']
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df['category']

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    X = pd.get_dummies(X, columns=categorical_features, dummy_na=False)

    scaler = MinMaxScaler()
    valid_num = [c for c in numerical_features if c in X.columns]
    X[valid_num] = scaler.fit_transform(X[valid_num])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded)

    return (X_train.to_numpy(dtype=np.float32), y_train), (X_test.to_numpy(dtype=np.float32), y_test), le.classes_, X_train.shape[1], {"train_size": len(X_train), "test_size": len(X_test)}

# ============================================================
# 3. Models and Evaluation
# ============================================================

def create_mlp_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(256, activation='relu'), Dropout(0.4),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(64, activation='relu'), Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def stratified_subset(X, y, size, seed=42):
    if size is None or size >= len(y):
        return X, y
    _, X_sub, _, y_sub = train_test_split(X, y, test_size=size, stratify=y, random_state=seed)
    return X_sub, y_sub


def evaluate_attack(model, X, y, attack_name, input_dim, num_classes, eps=0.1):
    loss_object = SparseCategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(
        model=model, nb_classes=num_classes, input_shape=(input_dim,),
        loss_object=loss_object, clip_values=(0, 1)
    )
    if attack_name == "PGD":
        attacker = ProjectedGradientDescent(estimator=classifier, eps=eps, max_iter=10, verbose=False)
    elif attack_name == "FGSM":
        attacker = FastGradientMethod(estimator=classifier, eps=eps)
    elif attack_name == "C&W":
        attacker = CarliniL2Method(classifier=classifier, confidence=0.0, max_iter=10, batch_size=32, verbose=False)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    X_adv = attacker.generate(x=X)
    pred = np.argmax(model.predict(X_adv, verbose=0), axis=1)
    return accuracy_score(y, pred), pred


def save_classification_report(model, X, y, class_names, dataset_name, model_name):
    pred = np.argmax(model.predict(X, verbose=0), axis=1)
    report = classification_report(y, pred, target_names=[str(c) for c in class_names], output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    out = os.path.join(RESULTS_DIR, f"per_class_report_{dataset_name}_{model_name}.csv")
    df_report.to_csv(out, index=True)
    return out

# ============================================================
# 4. Main Experiment
# ============================================================

if __name__ == "__main__":
    if DATASET == "UNSW_NB15":
        data_res = load_and_preprocess_unsw_nb15(UNSW_TRAIN_FILE, UNSW_TEST_FILE, binary=True, apply_smote=False)
        DATASET_NAME = "UNSW-NB15"
    elif DATASET == "NSL_KDD":
        data_res = load_and_preprocess_nsl_kdd(NSL_KDD_DIR)
        DATASET_NAME = "NSL-KDD"
    elif DATASET == "BOT_IOT":
        data_res = load_and_preprocess_bot_iot(BOT_IOT_FILE)
        DATASET_NAME = "Bot-IoT"
    else:
        raise ValueError("DATASET must be one of: UNSW_NB15, NSL_KDD, BOT_IOT")

    (X_train, y_train), (X_test, y_test), class_names, input_dim, meta = data_res
    num_classes = len(class_names)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Train shape: {X_train.shape}; Test shape: {X_test.shape}; Classes: {class_names}")

    all_rows = []
    table10_rows = []

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run + 1}/{NUM_RUNS} ---")
        set_seeds(RANDOM_SEED + run)

        print("Training Standard Baseline Model...")
        base_model = create_mlp_model(input_dim, num_classes)
        base_model.fit(
            X_train, y_train,
            epochs=EPOCHS_BASELINE,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        )

        print("Training TRADES Defended Model...")
        trades_model = create_mlp_model(input_dim, num_classes)
        loss_c = SparseCategoricalCrossentropy(from_logits=False)
        loss_r = KLDivergence()
        clf_trades = TensorFlowV2Classifier(model=trades_model, nb_classes=num_classes, input_shape=(input_dim,), loss_object=loss_c, clip_values=(0, 1))
        atk_trades = ProjectedGradientDescent(estimator=clf_trades, eps=ATTACK_EPSILON, max_iter=10, verbose=False)
        optimizer = Adam(0.001)
        ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(BATCH_SIZE)

        history_clean_loss = []
        history_robust_loss = []
        for epoch in range(EPOCHS_ADV):
            print(f"  > TRADES Epoch {epoch + 1}/{EPOCHS_ADV}")
            epoch_clean_loss_avg = tf.keras.metrics.Mean()
            epoch_robust_loss_avg = tf.keras.metrics.Mean()
            for x_batch, y_batch in ds:
                with tf.GradientTape() as tape:
                    x_adv = atk_trades.generate(x=x_batch.numpy())
                    logits_clean = trades_model(x_batch, training=True)
                    logits_adv = trades_model(x_adv, training=True)
                    l_clean = loss_c(y_batch, logits_clean)
                    l_robust = loss_r(tf.nn.softmax(logits_clean), tf.nn.softmax(logits_adv))
                    loss = l_clean + BETA * l_robust
                grads = tape.gradient(loss, trades_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, trades_model.trainable_variables))
                epoch_clean_loss_avg.update_state(l_clean)
                epoch_robust_loss_avg.update_state(l_robust)
            history_clean_loss.append(float(epoch_clean_loss_avg.result().numpy()))
            history_robust_loss.append(float(epoch_robust_loss_avg.result().numpy()))

        # Save training loss plot
        plt.figure(figsize=(8, 5))
        plt.plot(history_clean_loss, label="Clean CE Loss")
        plt.plot(history_robust_loss, label="Robust KL Loss", linestyle="--")
        plt.title(f"TRADES Training Dynamics on {DATASET_NAME}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"trades_loss_dynamics_{DATASET_NAME}.png"), dpi=300)
        plt.close()

        # Clean accuracies on full test set
        base_clean_pred = np.argmax(base_model.predict(X_test, verbose=0), axis=1)
        trades_clean_pred = np.argmax(trades_model.predict(X_test, verbose=0), axis=1)
        base_clean_acc = accuracy_score(y_test, base_clean_pred)
        trades_clean_acc = accuracy_score(y_test, trades_clean_pred)

        # Attack subsets
        X_pgd, y_pgd = stratified_subset(X_test, y_test, PGD_EVAL_SIZE, RANDOM_SEED)
        X_fgsm, y_fgsm = stratified_subset(X_test, y_test, FGSM_EVAL_SIZE, RANDOM_SEED + 1)
        X_cw, y_cw = stratified_subset(X_test, y_test, CW_EVAL_SIZE, RANDOM_SEED + 2)

        for model_name, model, clean_acc in [("Baseline", base_model, base_clean_acc), ("TRADES", trades_model, trades_clean_acc)]:
            pgd_acc, _ = evaluate_attack(model, X_pgd, y_pgd, "PGD", input_dim, num_classes, ATTACK_EPSILON)
            fgsm_acc, _ = evaluate_attack(model, X_fgsm, y_fgsm, "FGSM", input_dim, num_classes, ATTACK_EPSILON)
            cw_acc, _ = evaluate_attack(model, X_cw, y_cw, "C&W", input_dim, num_classes, ATTACK_EPSILON)
            all_rows.append({
                "Dataset": DATASET_NAME,
                "Run": run + 1,
                "Model": model_name,
                "Clean Accuracy": clean_acc,
                "PGD Robust Accuracy": pgd_acc,
                "FGSM Robust Accuracy": fgsm_acc,
                "C&W Robust Accuracy": cw_acc,
                "PGD Eval Samples": len(y_pgd),
                "FGSM Eval Samples": len(y_fgsm),
                "C&W Eval Samples": len(y_cw),
                "Epsilon": ATTACK_EPSILON
            })
            save_classification_report(model, X_test, y_test, class_names, DATASET_NAME, model_name)

        # TFLite conversion and size/latency for TRADES
        print("Converting TRADES model to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(trades_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = os.path.join(RESULTS_DIR, f"trades_model_{DATASET_NAME}.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        model_size_kb = os.path.getsize(tflite_path) / 1024

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        dummy_input = np.array(np.random.random_sample(input_details[0]["shape"]), dtype=np.float32)

        for _ in range(50):
            interpreter.set_tensor(input_details[0]["index"], dummy_input)
            interpreter.invoke()
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            interpreter.set_tensor(input_details[0]["index"], dummy_input)
            interpreter.invoke()
        end = time.perf_counter()
        avg_latency_ms = ((end - start) * 1000) / iterations
        throughput = 1000 / avg_latency_ms

        deploy_df = pd.DataFrame([{
            "Dataset": DATASET_NAME,
            "Model": "TRADES-TFLite",
            "TFLite Model Size (KB)": model_size_kb,
            "Average Latency (ms/packet)": avg_latency_ms,
            "Throughput (packets/sec)": throughput
        }])
        deploy_df.to_csv(os.path.join(RESULTS_DIR, f"deployment_metrics_{DATASET_NAME}.csv"), index=False)

    results_df = pd.DataFrame(all_rows)
    attack_out = os.path.join(RESULTS_DIR, f"attack_results_{DATASET_NAME}.csv")
    results_df.to_csv(attack_out, index=False)

    # Reviewer Table 10: TRADES row only, averaged across runs
    trades_rows = results_df[results_df["Model"] == "TRADES"]
    table10 = pd.DataFrame([{
        "Dataset": DATASET_NAME,
        "Samples Used": f"Official train/test split ({len(y_train)} train; {len(y_test)} test); attacks on stratified subsets: PGD={PGD_EVAL_SIZE}, FGSM={FGSM_EVAL_SIZE}, C&W={CW_EVAL_SIZE}",
        "No. of Classes": num_classes,
        "Clean Accuracy": trades_rows["Clean Accuracy"].mean(),
        "PGD Robust Accuracy": trades_rows["PGD Robust Accuracy"].mean(),
        "FGSM Robust Accuracy": trades_rows["FGSM Robust Accuracy"].mean(),
        "C&W Robust Accuracy": trades_rows["C&W Robust Accuracy"].mean(),
        "Notes": "Added cross-dataset benchmark using UNSW-NB15 binary normal-vs-attack classification."
    }])
    table10_path = os.path.join(RESULTS_DIR, "table10_cross_dataset_results.csv")
    table10.to_csv(table10_path, index=False)

    print("\nSaved outputs:")
    print(f"- {attack_out}")
    print(f"- {table10_path}")
    print(f"- {os.path.join(RESULTS_DIR, f'deployment_metrics_{DATASET_NAME}.csv')}")
    print("\nTable 10 row:")
    print(table10.to_string(index=False))
