# ===================================================================
# Part 1: Project Setup and All Imports
# ===================================================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Adversarial Robustness Toolbox (ART) imports
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier

# --- SET SEEDS FOR REPRODUCIBILITY ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# --- END OF SEED SETTING ---

print("All libraries imported successfully.")


# ===================================================================
# Part 2: Configuration
# ===================================================================

# --- Visual Style Configuration ---
BASELINE_COLOR = "royalblue"
ADVT_COLOR = "forestgreen"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --- TensorFlow Configuration ---
tf.config.set_visible_devices([], 'GPU')

# --- Experiment Parameters ---
ATTACK_EPSILON = 0.1  # Attack strength for evaluation attacks


# ===================================================================
# Part 3: Data Loading & Preprocessing 
# ===================================================================

def load_and_preprocess_nsl_kdd(data_path=''):
    """ Loads and preprocesses the NSL-KDD dataset. """
    print("Loading and preprocessing NSL-KDD dataset...")
    # (Code for NSL-KDD is complete and correct from previous version)
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
    df_train = pd.read_csv(data_path + 'KDDTrain+.txt', header=None, names=columns)
    df_test = pd.read_csv(data_path + 'KDDTest+.txt', header=None, names=columns)
    df_train.drop('difficulty', axis=1, inplace=True); df_test.drop('difficulty', axis=1, inplace=True)
    label_mapping = {
        'normal': 'normal', 'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 
        'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l',
        'warezclient': 'r2l', 'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l',
        'snmpguess': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r', 'httptunnel': 'u2r',
        'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
    }
    df_train['label'] = df_train['label'].map(label_mapping); df_test['label'] = df_test['label'].map(label_mapping)
    df_train.dropna(inplace=True); df_test.dropna(inplace=True)
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_full = pd.concat([df_train, df_test], axis=0)
    df_full_encoded = pd.get_dummies(df_full, columns=categorical_cols)
    df_train_encoded = df_full_encoded.iloc[:len(df_train)]; df_test_encoded = df_full_encoded.iloc[len(df_train):]
    label_encoder = LabelEncoder()
    X_train = df_train_encoded.drop('label', axis=1); y_train = label_encoder.fit_transform(df_train_encoded['label'])
    X_test = df_test_encoded.drop('label', axis=1); y_test = label_encoder.transform(df_test_encoded['label'])
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    print("NSL-KDD dataset loaded and preprocessed successfully.")
    return (X_train.astype(np.float32), y_train), (X_test.astype(np.float32), y_test), label_encoder.classes_, X_train.shape[1]

def load_and_preprocess_bot_iot(file_path, sample_size=200000):
    """ Loads the full Bot-IoT dataset and takes a stratified sample. """
    print(f"Loading and preprocessing Bot-IoT dataset from: {file_path}...")
    # (Code for Bot-IoT is complete and correct from previous version)
    try:
        df_full = pd.read_csv(file_path, low_memory=False)
        if sample_size > len(df_full): sample_size = len(df_full)
        df, _ = train_test_split(df_full, train_size=sample_size, stratify=df_full['category'], random_state=SEED)
        del df_full
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}"); return None, None, None, None
    cols_to_drop = ['pkSeqID', 'stime', 'flgs', 'saddr', 'daddr', 'sport', 'dport', 'attack', 'category', 'subcategory']
    X = df.drop(columns=cols_to_drop, errors='ignore'); y = df['category']
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    X = pd.get_dummies(X, columns=categorical_features, dummy_na=False)
    scaler = MinMaxScaler()
    valid_numerical_features = [col for col in numerical_features if col in X.columns]
    X[valid_numerical_features] = scaler.fit_transform(X[valid_numerical_features])
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded)
    print("Bot-IoT dataset loaded and preprocessed successfully.")
    return (X_train.to_numpy(dtype=np.float32), y_train), (X_test.to_numpy(dtype=np.float32), y_test), label_encoder.classes_, X_train.shape[1]


# ===================================================================
# Part 4: Model Architecture and Training 
# ===================================================================

def create_powerful_mlp(input_shape, num_classes):
    """ Creates a powerful Keras MLP model for complex datasets. """
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(256, activation='relu'), Dropout(0.4),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(64, activation='relu'), Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)


# ===================================================================
# Part 5: Main Experiment Workflow
# ===================================================================

# --- CHOOSE YOUR DATASET ---
USE_BOT_IOT = False # Set to True for Bot-IoT, False for NSL-KDD

if USE_BOT_IOT:
    file_path = 'data/Bot-IoT.csv' 
    (X_train, y_train), (X_test, y_test), class_names, input_shape_dim = load_and_preprocess_bot_iot(file_path)
    DATASET_NAME = "Bot-IoT"
else:
    (X_train, y_train), (X_test, y_test), class_names, input_shape_dim = load_and_preprocess_nsl_kdd(data_path='data/')
    DATASET_NAME = "NSL-KDD"

if X_train is None:
    print("\nData loading failed. Exiting script.")
else:
    num_classes = len(class_names)
    print(f"\nDataset is ready: {DATASET_NAME}. Input shape: {input_shape_dim}, Classes: {num_classes}")

    # --- Train Baseline Model ---
    print("\n--- Training Baseline Model ---")
    baseline_model = create_powerful_mlp(input_shape_dim, num_classes)
    baseline_model.fit(
        X_train, y_train, epochs=100, batch_size=256,
        validation_split=0.2, callbacks=[early_stopper], verbose=1
    )

    # --- Train Defended Model (via TRADES) ---
    print("\n--- Training Defended Model (via TRADES) ---")
    trades_model = create_powerful_mlp(input_shape_dim, num_classes)
    EPOCHS_TRADES = 30; BATCH_SIZE_TRADES = 256; LEARNING_RATE_TRADES = 0.001; BETA = 1.0
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train), seed=SEED).batch(BATCH_SIZE_TRADES)
    optimizer = Adam(learning_rate=LEARNING_RATE_TRADES)
    loss_fn_clean = SparseCategoricalCrossentropy(from_logits=False)
    loss_fn_robust = KLDivergence()
    art_classifier_for_trades = TensorFlowV2Classifier(
        model=trades_model, nb_classes=num_classes, input_shape=(input_shape_dim,),
        loss_object=loss_fn_clean, clip_values=(0, 1)
    )
    pgd_attack_for_trades = ProjectedGradientDescent(estimator=art_classifier_for_trades, eps=0.1, eps_step=0.01, max_iter=10)
    for epoch in range(EPOCHS_TRADES):
        print(f"Epoch {epoch + 1}/{EPOCHS_TRADES}")
        epoch_loss_avg = tf.keras.metrics.Mean()
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                x_batch_adv = pgd_attack_for_trades.generate(x=x_batch.numpy())
                logits_clean = trades_model(x_batch, training=True)
                logits_adv = trades_model(x_batch_adv, training=True)
                loss_natural = loss_fn_clean(y_batch, logits_clean)
                loss_robust = loss_fn_robust(tf.nn.softmax(logits_clean), tf.nn.softmax(logits_adv))
                total_loss = loss_natural + (BETA * loss_robust)
            grads = tape.gradient(total_loss, trades_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, trades_model.trainable_variables))
            epoch_loss_avg.update_state(total_loss)
        print(f"  Average Loss: {epoch_loss_avg.result():.4f}")
    print("TRADES training complete.")

    # ===================================================================
    # Part 6: Systematic Evaluation
    # ===================================================================
    print("\n" + "="*50); print("Starting Systematic Evaluation"); print("="*50)
    models_to_test = { "Baseline": baseline_model, "TRADES": trades_model }
    results = {}
    for model_name, model in models_to_test.items():
        print(f"\n--- Evaluating Model: {model_name} ---")
        results[model_name] = {}
        art_classifier = TensorFlowV2Classifier(
            model=model, nb_classes=num_classes, input_shape=(input_shape_dim,),
            loss_object=SparseCategoricalCrossentropy(), clip_values=(0, 1)
        )
        # Test on Clean Data
        y_pred_clean = np.argmax(model.predict(X_test, verbose=0), axis=1)
        results[model_name]["Clean"] = classification_report(y_test, y_pred_clean, target_names=class_names, output_dict=True, zero_division=0)
        print(f"Accuracy on Clean Data: {results[model_name]['Clean']['accuracy']:.4f}")
        # Test on PGD Attack
        attack_pgd = ProjectedGradientDescent(estimator=art_classifier, eps=ATTACK_EPSILON, max_iter=10)
        X_test_adv_pgd = attack_pgd.generate(x=X_test)
        y_pred_pgd = np.argmax(model.predict(X_test_adv_pgd, verbose=0), axis=1)
        results[model_name]["PGD"] = classification_report(y_test, y_pred_pgd, target_names=class_names, output_dict=True, zero_division=0)
        print(f"Accuracy under PGD Attack (eps={ATTACK_EPSILON}): {results[model_name]['PGD']['accuracy']:.4f}")
    print("\nExperiment finished.")
    
    # ===================================================================
    # Part 7: Visualizations
    # ===================================================================
    print("Generating visualizations...")
    # --- Figure 1: Overall Accuracy Comparison ---
    plot_data = []
    for model_name, conditions in results.items():
        for condition_name, report in conditions.items():
            plot_data.append({'Model': model_name, 'Condition': condition_name, 'Accuracy': report['accuracy']})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Condition', y='Accuracy', hue='Model', data=df_plot, palette=[BASELINE_COLOR, ADVT_COLOR])
    plt.title(f'Overall Model Accuracy on {DATASET_NAME}', fontsize=16)
    plt.xlabel('Test Data Condition'); plt.ylabel('Accuracy'); plt.ylim(0, 1.0)
    plt.savefig(f'{DATASET_NAME}_overall_accuracy.png', dpi=300); plt.show()

    # --- Figure 2: Per-Class F1-Score Bar Chart (Under PGD Attack) ---
    f1_pgd_data = []
    for model_name, report in results.items():
        for class_name in class_names:
            f1_pgd_data.append({
                'Model': model_name,
                'Class': class_name,
                'F1-Score': report['PGD'][class_name]['f1-score']
            })
    df_f1_pgd = pd.DataFrame(f1_pgd_data)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='F1-Score', hue='Model', data=df_f1_pgd, palette=[BASELINE_COLOR, ADVT_COLOR])
    plt.title(f'Per-Class F1-Score under PGD Attack on {DATASET_NAME}', fontsize=16)
    plt.xlabel('Attack Class'); plt.ylabel('F1-Score'); plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(f'{DATASET_NAME}_f1_scores_pgd.png', dpi=300); plt.show()

    # --- Figure 3: Accuracy vs. Robustness Trade-off Plot ---
    tradeoff_data = []
    for model_name, reports in results.items():
        tradeoff_data.append({
            'Model': model_name,
            'Robustness (PGD Accuracy)': reports['PGD']['accuracy'],
            'Standard Accuracy (Clean)': reports['Clean']['accuracy']
        })
    df_tradeoff = pd.DataFrame(tradeoff_data)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_tradeoff, x='Robustness (PGD Accuracy)', y='Standard Accuracy (Clean)', hue='Model', palette=[BASELINE_COLOR, ADVT_COLOR], s=200, style='Model', markers=['X', 'o'])
    plt.title(f'Accuracy vs. Robustness Trade-off on {DATASET_NAME}', fontsize=16)
    plt.xlim(0, 1.05); plt.ylim(0, 1.05)
    plt.grid(True); plt.tight_layout()
    plt.savefig(f'{DATASET_NAME}_tradeoff_plot.png', dpi=300); plt.show()

    # ===================================================================
    # Part 8: IoT Deployment Preparation & Model Complexity
    # ===================================================================
    print("\n--- Preparing final model for IoT deployment ---")
    deployment_model = trades_model
    deployment_model.save(f'defended_{DATASET_NAME}_model.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(deployment_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(f'defended_{DATASET_NAME}_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model successfully converted to TensorFlow Lite format.")

    print("\n--- Analyzing Model Complexity ---")
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
    forward_pass = tf.function(deployment_model.call, input_signature=[tf.TensorSpec(shape=(1,) + deployment_model.input_shape[1:])])
    graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
    flops = graph_info.total_float_ops
    print(f"Model Complexity: {flops / 1e6:.2f} MFLOPs")