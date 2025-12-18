"""
Hardening the IoT Edge: A TRADES-based Approach for Robust Network Intrusion Detection

This script implements the experimental framework for evaluating the robustness of 
Deep Learning-based NIDS against adversarial attacks. It compares a standard baseline 
model, a PGD Adversarially Trained (PGD-AT) model, and a TRADES-defended model 
across NSL-KDD and Bot-IoT datasets.

Key Features:
1. Data Preprocessing with SMOTE for class balancing.
2. Implementation of TRADES loss function.
3. Comparative evaluation against PGD, FGSM, and C&W attacks.
4. Deployment analysis using TensorFlow Lite (Latency/Throughput).

Author: [Your Name]
Date: 2025
"""

# ===================================================================
# 1. Setup and Imports
# ===================================================================
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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Adversarial Robustness Toolbox imports
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import TensorFlowV2Classifier

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# For reproducibility
def set_seeds(seed_value=42):
    """Sets random seeds for reproducibility across numpy, python, and tensorflow."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

set_seeds(42)

# ===================================================================
# 2. Configuration Parameters
# ===================================================================

# Experiment Settings
NUM_RUNS = 1              
EPOCHS_BASELINE = 15      
EPOCHS_ADV = 5            
BATCH_SIZE = 512          
ATTACK_EPSILON = 0.1      

# Dataset Selection
# Set to True for Bot-IoT, False for NSL-KDD
USE_BOT_IOT = False  

# Paths
# Note: Update this path to point to your local dataset directory
DATA_PATH_ROOT = "C:/Users/POP/New folder/data/"
BOT_IOT_FILE = "C:/Users/POP/New folder/reduced_data_4.csv"

# Visualization Settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
tf.config.set_visible_devices([], 'GPU') 

print("Configuration loaded. Libraries imported successfully.")

# ===================================================================
# 3. Data Loading & Preprocessing
# ===================================================================

def load_and_preprocess_nsl_kdd(data_path):
    """
    Loads, encodes, scales, and balances the NSL-KDD dataset.
    """
    print(f"Loading NSL-KDD dataset from {data_path}...")
    
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

    if not os.path.exists(train_path):
        print(f"Error: Data file not found at {train_path}")
        return None, None, None, None

    df_train = pd.read_csv(train_path, header=None, names=columns)
    df_test = pd.read_csv(test_path, header=None, names=columns)

    
    df_train.drop('difficulty', axis=1, inplace=True)
    df_test.drop('difficulty', axis=1, inplace=True)
    
    # Map specific attack types to broad categories
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
    
    # One-Hot Encoding
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_full = pd.concat([df_train, df_test], axis=0)
    df_full_encoded = pd.get_dummies(df_full, columns=categorical_cols)
    
    df_train_encoded = df_full_encoded.iloc[:len(df_train)]
    df_test_encoded = df_full_encoded.iloc[len(df_train):]
    
    # Label Encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train_encoded['label'])
    y_test = label_encoder.transform(df_test_encoded['label'])
    
    X_train = df_train_encoded.drop('label', axis=1)
    X_test = df_test_encoded.drop('label', axis=1)
    
    # Class Balancing via SMOTE
    print(f"  [Pre-SMOTE] Training set size: {X_train.shape[0]}")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  [Post-SMOTE] Training set size: {X_train.shape[0]}")
    
    # Feature Scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (X_train.astype(np.float32), y_train), (X_test.astype(np.float32), y_test), label_encoder.classes_, X_train.shape[1]

def load_and_preprocess_bot_iot(file_path, sample_size=200000):
    """
    Loads, preprocesses, and splits the Bot-IoT dataset.
    """
    print(f"Loading Bot-IoT dataset from {file_path}...")
    
    try:
        df_full = pd.read_csv(file_path, low_memory=False)
        # Stratified sampling for efficiency
        if sample_size > len(df_full): 
            sample_size = len(df_full)
        df, _ = train_test_split(df_full, train_size=sample_size, stratify=df_full['category'], random_state=42)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None

    # Drop irrelevant identifier columns
    cols_to_drop = ['pkSeqID', 'stime', 'flgs', 'saddr', 'daddr', 'sport', 'dport', 'attack', 'category', 'subcategory']
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df['category']
    
    # Encoding
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    X = pd.get_dummies(X, columns=categorical_features, dummy_na=False)
    
    # Scaling
    scaler = MinMaxScaler()
    valid_numerical_features = [col for col in numerical_features if col in X.columns]
    X[valid_numerical_features] = scaler.fit_transform(X[valid_numerical_features])
    
    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    return (X_train.to_numpy(dtype=np.float32), y_train), (X_test.to_numpy(dtype=np.float32), y_test), label_encoder.classes_, X_train.shape[1]

# ===================================================================
# 4. Model Definition
# ===================================================================

def create_mlp_model(input_shape, num_classes):
    """
    Creates a standardized Multi-Layer Perceptron (MLP) for NIDS.
    """
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(256, activation='relu'), Dropout(0.4),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(64, activation='relu'), Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ===================================================================
# 4.1  Analysis Functions
# ===================================================================

def plot_epsilon_curve(model, X_eval, y_eval, input_dim, num_classes, dataset_name):
    """Plots accuracy degradation across increasing epsilon values."""
    epsilons = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    accuracies = []
    
    print("\n--- Generating Epsilon Sensitivity Curve ---")
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(model=model, nb_classes=num_classes, 
                                        input_shape=(input_dim,), loss_object=loss_object, 
                                        clip_values=(0, 1))
    
    for eps in epsilons:
        if eps == 0.0:
            acc = accuracy_score(y_eval, np.argmax(model.predict(X_eval, verbose=0), axis=1))
        else:
            attacker = ProjectedGradientDescent(estimator=classifier, eps=eps, max_iter=10, verbose=False)
            x_adv = attacker.generate(X_eval)
            acc = accuracy_score(y_eval, np.argmax(model.predict(x_adv, verbose=0), axis=1))
        
        accuracies.append(acc)
        print(f"Epsilon: {eps} -> Accuracy: {acc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, accuracies, marker='o', linewidth=2, label='TRADES Model')
    plt.title(f'Robustness vs. Attack Strength (Epsilon) on {dataset_name}')
    plt.xlabel('Perturbation Budget (Epsilon)')
    plt.ylabel('Model Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('epsilon_curve.png')
    plt.show()

def plot_adversarial_confusion_matrix(model, X_eval, y_eval, input_dim, num_classes, class_names):
    """Plots confusion matrix specifically for PGD-attacked data."""
    print("\n--- Generating Adversarial Confusion Matrix ---")
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(model=model, nb_classes=num_classes, 
                                        input_shape=(input_dim,), loss_object=loss_object, 
                                        clip_values=(0, 1))
    
    # Generate Attack
    attacker = ProjectedGradientDescent(estimator=classifier, eps=0.1, max_iter=10, verbose=False)
    x_adv = attacker.generate(X_eval)
    
    # Predictions
    y_pred = np.argmax(model.predict(x_adv, verbose=0), axis=1)
    
    # Generate Matrix
    cm = confusion_matrix(y_eval, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix under PGD Attack (TRADES Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('adv_confusion_matrix.png')
    plt.show()

def plot_tsne_clusters(model, X_eval, y_eval, input_dim, num_classes):
    """Visualizes the latent space of Clean vs Adversarial examples using t-SNE."""
    print("\n--- Generating t-SNE Latent Space Visualization ---")
    
    # Create feature extractor (Penultimate Layer)
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    
    # Get Clean Features
    features_clean = feature_extractor.predict(X_eval, verbose=0)
    
    # Get Adversarial Features
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(model=model, nb_classes=num_classes, 
                                        input_shape=(input_dim,), loss_object=loss_object, 
                                        clip_values=(0, 1))
    attacker = ProjectedGradientDescent(estimator=classifier, eps=0.1, max_iter=10, verbose=False)
    x_adv = attacker.generate(X_eval)
    features_adv = feature_extractor.predict(x_adv, verbose=0)
    
    # Combine for t-SNE
    combined_features = np.vstack([features_clean, features_adv])
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(combined_features)
    
    # Split back
    half = len(features_clean)
    tsne_clean = tsne_results[:half]
    tsne_adv = tsne_results[half:]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_clean[:, 0], tsne_clean[:, 1], c='blue', alpha=0.5, label='Clean Samples')
    plt.scatter(tsne_adv[:, 0], tsne_adv[:, 1], c='red', alpha=0.5, marker='x', label='Adversarial Samples')
    
    plt.title('Latent Space (t-SNE): Clean vs Adversarial Samples')
    plt.legend()
    plt.savefig('tsne_analysis.png')
    plt.show()

# ===================================================================
# 5. Main Experiment Execution
# ===================================================================

if __name__ == "__main__":
    
    # --- Load Data ---
    if USE_BOT_IOT:
        data_res = load_and_preprocess_bot_iot(BOT_IOT_FILE)
        DATASET_NAME = "Bot-IoT"
    else:
        data_res = load_and_preprocess_nsl_kdd(DATA_PATH_ROOT)
        DATASET_NAME = "NSL-KDD"

    X_train, y_train = data_res[0]
    X_test, y_test = data_res[1]
    class_names = data_res[2]
    input_dim = data_res[3]

    if X_train is not None:
        num_classes = len(class_names)
        
        # Dictionary to store performance metrics across runs
        stats = {
            "Baseline": {"Clean": [], "PGD": []},
            "TRADES":   {"Clean": [], "PGD": []}
        }
        if DATASET_NAME == "NSL-KDD":
            stats["PGD-AT"] = {"Clean": [], "PGD": []}

        print(f"\nStarting experiment on {DATASET_NAME} with {NUM_RUNS} run(s)...")

        for run in range(NUM_RUNS):
            print(f"\n--- Run {run + 1}/{NUM_RUNS} ---")
            set_seeds(42 + run)
            
            # -------------------------------------------------------
            # A. Train Baseline Model
            # -------------------------------------------------------
            print("Training Standard Baseline Model...")
            base_model = create_mlp_model(input_dim, num_classes)
            base_model.fit(
                X_train, y_train, 
                epochs=EPOCHS_BASELINE, 
                batch_size=BATCH_SIZE, 
                validation_split=0.2, 
                verbose=0, 
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
            )
            
            # -------------------------------------------------------
            # B. Train PGD-AT Model (Optional: NSL-KDD Only)
            # -------------------------------------------------------
            pgd_at_model = None
            if DATASET_NAME == "NSL-KDD":
                print("Training PGD Adversarial Training (PGD-AT) Model...")
                pgd_at_model = create_mlp_model(input_dim, num_classes)
                loss_fn = SparseCategoricalCrossentropy()
                
                # Wrap model for ART
                clf_at = TensorFlowV2Classifier(model=pgd_at_model, nb_classes=num_classes, input_shape=(input_dim,), loss_object=loss_fn, clip_values=(0,1))
                atk_at = ProjectedGradientDescent(estimator=clf_at, eps=ATTACK_EPSILON, max_iter=10)
                
                # Custom Training Loop for PGD-AT
                ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(BATCH_SIZE)
                optimizer = Adam(0.001)
                
                for epoch in range(EPOCHS_ADV): 
                    print(f"  > PGD-AT Epoch {epoch+1}/{EPOCHS_ADV}")
                    for x_batch, y_batch in ds:
                        # Generate adversarial examples on the fly
                        x_adv = atk_at.generate(x=x_batch.numpy())
                        with tf.GradientTape() as tape:
                            loss = loss_fn(y_batch, pgd_at_model(x_adv, training=True))
                        grads = tape.gradient(loss, pgd_at_model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, pgd_at_model.trainable_variables))

            # -------------------------------------------------------
            # C. Train TRADES Model
            # -------------------------------------------------------
            print("Training TRADES Defended Model...")
            trades_model = create_mlp_model(input_dim, num_classes)
            loss_c = SparseCategoricalCrossentropy(from_logits=False)
            loss_r = KLDivergence()
            
            # Wrap model for ART
            clf_trades = TensorFlowV2Classifier(model=trades_model, nb_classes=num_classes, input_shape=(input_dim,), loss_object=loss_c, clip_values=(0,1))
            atk_trades = ProjectedGradientDescent(estimator=clf_trades, eps=ATTACK_EPSILON, max_iter=10)
            
            # Custom Training Loop for TRADES
            ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(BATCH_SIZE)
            optimizer = Adam(0.001)
            
            # Metric trackers for plotting
            history_clean_loss = []
            history_robust_loss = []
            
            for epoch in range(EPOCHS_ADV):
                print(f"  > TRADES Epoch {epoch+1}/{EPOCHS_ADV}")
                
                # Epoch accumulators
                epoch_clean_loss_avg = tf.keras.metrics.Mean()
                epoch_robust_loss_avg = tf.keras.metrics.Mean()

                for x_batch, y_batch in ds:
                    with tf.GradientTape() as tape:
                        # 1. Generate adversarial example
                        x_adv = atk_trades.generate(x=x_batch.numpy())
                        
                        # 2. Forward pass clean and adv
                        logits_clean = trades_model(x_batch, training=True)
                        logits_adv = trades_model(x_adv, training=True)
                        
                        # 3. Calculate TRADES loss: Clean Acc + Beta * Consistency
                        l_clean = loss_c(y_batch, logits_clean)
                        l_robust = loss_r(tf.nn.softmax(logits_clean), tf.nn.softmax(logits_adv))
                        loss = l_clean + l_robust
                    
                    grads = tape.gradient(loss, trades_model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, trades_model.trainable_variables))
                    
                    # Update trackers
                    epoch_clean_loss_avg.update_state(l_clean)
                    epoch_robust_loss_avg.update_state(l_robust)
                
                # Store epoch averages
                history_clean_loss.append(epoch_clean_loss_avg.result().numpy())
                history_robust_loss.append(epoch_robust_loss_avg.result().numpy())

            # Plot Training Dynamics
            plt.figure(figsize=(10, 5))
            plt.plot(history_clean_loss, label='Standard Accuracy Loss')
            plt.plot(history_robust_loss, label='Robustness (KL) Loss', linestyle='--')
            plt.title('TRADES Training Dynamics: Accuracy vs. Stability')
            plt.xlabel('Epochs')
            plt.ylabel('Loss Value')
            plt.legend()
            plt.savefig('trades_loss_dynamics.png')
            plt.show()
            
            # -------------------------------------------------------
            # D. Evaluation Phase
            # -------------------------------------------------------
            models = {"Baseline": base_model, "TRADES": trades_model}
            if pgd_at_model: models["PGD-AT"] = pgd_at_model
            
            # Create evaluation subset for efficiency
            idx = np.random.choice(len(X_test), 500, replace=False)
            X_eval, y_eval = X_test[idx], y_test[idx]
            
            for name, model in models.items():
                # 1. Standard Accuracy
                clean_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                clean_acc = accuracy_score(y_test, clean_pred)
                stats[name]["Clean"].append(clean_acc)
                
                # 2. Robust Accuracy (PGD)
                # Create a fresh classifier wrapper for evaluation
                eval_clf = TensorFlowV2Classifier(model=model, nb_classes=num_classes, input_shape=(input_dim,), loss_object=loss_c, clip_values=(0,1))
                eval_pgd = ProjectedGradientDescent(estimator=eval_clf, eps=ATTACK_EPSILON, max_iter=10, verbose=False)
                
                x_test_adv = eval_pgd.generate(X_eval)
                robust_pred = np.argmax(model.predict(x_test_adv, verbose=0), axis=1)
                robust_acc = accuracy_score(y_eval, robust_pred)
                
                stats[name]["PGD"].append(robust_acc)
        
        # ===================================================================
        # 6. Reporting Results
        # ===================================================================
        print("\n" + "="*40)
        print(" FINAL STATISTICAL REPORT ")
        print("="*40)
        print(f"{'Model':<15} | {'Clean Accuracy':<20} | {'Robust Accuracy (PGD)':<20}")
        print("-" * 65)
        
        for name, m in stats.items():
            cm, cs = np.mean(m["Clean"]), np.std(m["Clean"])
            pm, ps = np.mean(m["PGD"]), np.std(m["PGD"])
            print(f"{name:<15} | {cm:.4f} +/- {cs:.4f}    | {pm:.4f} +/- {ps:.4f}")

        # ===================================================================
        # 7. Expanded Attack Evaluation (FGSM & C&W)
        # ===================================================================
        print("\n" + "="*40)
        print(" CROSS-ATTACK GENERALIZATION (FGSM & C&W) ")
        print("="*40)
        
        # Subset for attack evaluation
        eval_size = 200 
        idx = np.random.choice(len(X_test), eval_size, replace=False)
        X_sub, y_sub = X_test[idx], y_test[idx]
        
        eval_models = {"Baseline": base_model, "TRADES": trades_model}
        if 'pgd_at_model' in locals() and pgd_at_model is not None:
            eval_models["PGD-AT"] = pgd_at_model
            
        loss_fn = SparseCategoricalCrossentropy()

        for name, model in eval_models.items():
            print(f"\n--- Evaluating {name} ---")
            classifier = TensorFlowV2Classifier(model=model, nb_classes=num_classes, input_shape=(input_dim,), loss_object=loss_fn, clip_values=(0, 1))

            # FGSM Evaluation
            attacker_fgsm = FastGradientMethod(classifier, eps=ATTACK_EPSILON)
            x_fgsm = attacker_fgsm.generate(x=X_sub)
            acc_fgsm = accuracy_score(y_sub, np.argmax(model.predict(x_fgsm, verbose=0), axis=1))
            print(f"  FGSM Accuracy: {acc_fgsm:.4f}")

            # C&W Evaluation
            attacker_cw = CarliniL2Method(classifier, confidence=0.0, max_iter=10, batch_size=32, verbose=False)
            x_cw = attacker_cw.generate(x=X_sub)
            acc_cw = accuracy_score(y_sub, np.argmax(model.predict(x_cw, verbose=0), axis=1))
            print(f"  C&W Accuracy:  {acc_cw:.4f}")

        # ===================================================================
        # 8. Deployment Feasibility Analysis
        # ===================================================================
        print("\n" + "="*40)
        print(" EDGE DEPLOYMENT ANALYSIS (LATENCY) ")
        print("="*40)
        
        # Convert TRADES model to TFLite with Optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(trades_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Setup Interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        dummy_input = np.array(np.random.random_sample(input_details[0]['shape']), dtype=np.float32)
        
        # Warmup phase
        for _ in range(50): 
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        # Measurement phase
        num_iterations = 1000
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        end_time = time.perf_counter()
        
        avg_latency_ms = ((end_time - start_time) * 1000) / num_iterations
        throughput = 1000 / avg_latency_ms
        
        print(f"TRADES TFLite Latency: {avg_latency_ms:.4f} ms/packet")
        print(f"Projected Throughput:  {throughput:.2f} packets/sec")
        
        if avg_latency_ms < 1.0:
            print(">> Verdict: Suitable for Real-Time Edge IoT Applications")
        else:
            print(">> Verdict: Suitable for Near-Real-Time IoT Applications")
            
        print("\n" + "="*40)
        print(" GENERATING ENHANCED ANALYTICAL VISUALIZATIONS ")
        print("="*40)
        
        # Use a fresh subset for visualization
        plot_idx = np.random.choice(len(X_test), 200, replace=False)
        X_plot, y_plot = X_test[plot_idx], y_test[plot_idx]

        # 1. Epsilon Curve
        plot_epsilon_curve(trades_model, X_plot, y_plot, input_dim, num_classes, DATASET_NAME)

        # 2. Adversarial Confusion Matrix
        plot_adversarial_confusion_matrix(trades_model, X_plot, y_plot, input_dim, num_classes, class_names)
        
        # 3. t-SNE Visualization
        plot_tsne_clusters(trades_model, X_plot, y_plot, input_dim, num_classes)
        
        print("\nExperiment and Enhanced Analysis Complete.")
