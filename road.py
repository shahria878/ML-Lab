import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Importing Classifiers as per requirements
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

print("==================================================")
print("[Step 1/4] Loading and Preprocessing PVS 1 via Paper Methodology...")

# Directory Configuration
root_dir = os.path.join(os.path.dirname(__file__), "PVS 1")
filenames = {
    "left_gps_mpu": "dataset_gps_mpu_left.csv",
    "labels": "dataset_labels.csv",
}

all_features = []


# --- 1. BUTTERWORTH HIGH-PASS FILTER CONFIGURATION (Section 3.2.3) ---
def butter_highpass_filter(data, cutoff=2.0, fs=50.0, order=11):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)


try:
    sensor_path = os.path.join(root_dir, filenames["left_gps_mpu"])
    label_path = os.path.join(root_dir, filenames["labels"])

    df_sensor = pd.read_csv(sensor_path)
    df_label = pd.read_csv(label_path)

    # --- 2. MAPPING LABELS TO NEW CATEGORIES (Bad, Regular, Good) ---
    if "bad_road_left" in df_label.columns:
        conditions = [
            (df_label["bad_road_left"] == 1) | (df_label["bad_road_right"] == 1),
            (df_label["regular_road_left"] == 1)
            | (df_label["regular_road_right"] == 1),
            (df_label["good_road_left"] == 1)
            | (df_label["good_road_right"] == 1),
        ]
        # আপনার রিকোয়ারমেন্ট অনুযায়ী ক্লাস নেম পরিবর্তন করা হলো
        classes = ["Bad", "Regular", "Good"]
        df_label["road_type"] = np.select(conditions, classes, default="Good")
    else:
        road_classes = ["dirt_road", "cobblestone_road", "asphalt_road"]
        conditions = [df_label[r] == 1 for r in road_classes]
        classes = ["Bad", "Regular", "Good"]
        df_label["road_type"] = np.select(conditions, classes, default="Good")

    # --- 3. SIGNAL FILTERING (Section 3.2.3) ---
    acc_x = df_sensor["acc_x_dashboard"].values
    acc_y = df_sensor["acc_y_dashboard"].values
    acc_z = df_sensor["acc_z_dashboard"].values

    fs = 50.0
    filtered_x = butter_highpass_filter(acc_x, cutoff=2.0, fs=fs, order=11)
    filtered_y = butter_highpass_filter(acc_y, cutoff=2.0, fs=fs, order=11)
    filtered_z = butter_highpass_filter(acc_z, cutoff=2.0, fs=fs, order=11)

    # --- 4. SLIDING WINDOW SEGMENTATION & FEATURE EXTRACTION ---
    window_size = 64
    overlap = 32

    for i in range(0, len(df_sensor) - window_size, overlap):
        w_x = filtered_x[i : i + window_size]
        w_y = filtered_y[i : i + window_size]
        w_z = filtered_z[i : i + window_size]

        w_speed = df_sensor["speed"].iloc[i : i + window_size].mean()
        w_label = df_label["road_type"].iloc[i]

        if w_speed >= 2.77:
            rms_x = np.sqrt(np.mean(w_x**2))
            rms_y = np.sqrt(np.mean(w_y**2))
            rms_z = np.sqrt(np.mean(w_z**2))

            xz_ratio = rms_x / (rms_z + 1e-6)

            mean_z = np.mean(w_z)
            std_z = np.std(w_z)
            peak_z = np.max(np.abs(w_z))
            var_z = np.var(w_z)

            fft_z = np.abs(np.fft.fft(w_z))[: window_size // 2]
            fft_mean = np.mean(fft_z)
            fft_max = np.max(fft_z)
            fft_std = np.std(fft_z)

            all_features.append(
                {
                    "mean_z": mean_z,
                    "std_z": std_z,
                    "rms_x": rms_x,
                    "rms_y": rms_y,
                    "rms_z": rms_z,
                    "peak_z": peak_z,
                    "variance_z": var_z,
                    "xz_ratio": xz_ratio,
                    "fft_mean": fft_mean,
                    "fft_max": fft_max,
                    "fft_std": fft_std,
                    "speed_kmh": w_speed * 3.6,
                    "total_vibration": np.sum(np.abs(w_z)),
                    "road_target": w_label,
                }
            )

except FileNotFoundError:
    print(f"\n[Error] Dataset files missing inside target directory: {root_dir}")
    exit()

df_ml = pd.DataFrame(all_features)
feature_cols = [
    "mean_z",
    "std_z",
    "rms_x",
    "rms_y",
    "rms_z",
    "peak_z",
    "variance_z",
    "xz_ratio",
    "fft_mean",
    "fft_max",
    "fft_std",
]
X = df_ml[feature_cols]
y = df_ml["road_target"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print("Data processing ready. Real windowed samples generated:", len(df_ml))
print("==================================================\n")

# --- 5. MODEL TRAINING AND COMPARATIVE PRECISION REPORT ---
print(
    "[Step 2/4] Executing parallel model benchmarking with Precision/Recall Metrics..."
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=200, depth=5, learning_rate=0.1, random_state=42, verbose=0
    ),
}

accuracy_results = {}
trained_models = {}

for name, model in models.items():
    print("\n" + "-" * 60)
    print(f" 🚀 Training & Detailed Evaluation for Model: {name}")
    print("-" * 60)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions) * 100
    accuracy_results[name] = acc
    trained_models[name] = model

    # এখানে এখন রিপোর্ট 'Bad', 'Regular', 'Good' ফরম্যাটে প্রিন্ট হবে
    print(classification_report(y_test, predictions, target_names=le.classes_))

print("\n" + "=" * 60)
print("📊 OVERALL PERFORMANCE SUMMARY (ACCURACY COMPARISON)")
print("=" * 60)
print(f"{'Algorithm Model Classifier':<25} | {'Evaluation Accuracy %':<20}")
print("--------------------------------------------------")
for model_name, score in accuracy_results.items():
    print(f"{model_name:<25} | {score:.2f}%")
print("==================================================\n")

best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = trained_models[best_model_name]
print(
    f"⭐ Automated Selection: **{best_model_name}** is selected for real-time inference.\n"
)


# --- 6. USER INTERACTIVE REAL-TIME INFERENCE SIMULATOR ---
print("--- [Step 3/4] Test New Road Data (Continuous Window Simulation) ---")
print("To predict accurately, please provide a sample stream block:")

try:
    user_lat = float(input("1. Enter current Latitude (e.g., 30.5564): "))
    user_lon = float(input("2. Enter current Longitude (e.g., 120.1542): "))
    user_speed = float(input("3. Enter vehicle current Speed in km/h (e.g., 35): "))

    print("\n--- Simulate a 64-Sample Data Stream Block ---")
    u_acc_x_base = float(input("4. Base Acceleration X (Lateral sway, e.g., 0.1): "))
    u_acc_y_base = float(input("5. Base Acceleration Y (Longitudinal, e.g., 0.2): "))
    u_acc_z_base = float(
        input("6. Base Acceleration Z (Vertical impact shock, e.g., 1.5): ")
    )

    np.random.seed(42)
    sim_x = np.random.normal(u_acc_x_base, 0.2, 64)
    sim_y = np.random.normal(u_acc_y_base, 0.2, 64)
    sim_z = np.random.normal(u_acc_z_base, 0.5, 64)

    f_sim_x = butter_highpass_filter(sim_x, cutoff=2.0, fs=fs, order=11)
    f_sim_y = butter_highpass_filter(sim_y, cutoff=2.0, fs=fs, order=11)
    f_sim_z = butter_highpass_filter(sim_z, cutoff=2.0, fs=fs, order=11)

    u_rms_x = np.sqrt(np.mean(f_sim_x**2))
    u_rms_y = np.sqrt(np.mean(f_sim_y**2))
    u_rms_z = np.sqrt(np.mean(f_sim_z**2))

    user_fft = np.abs(np.fft.fft(f_sim_z))[:32]

    user_features = pd.DataFrame(
        [
            {
                "mean_z": np.mean(f_sim_z),
                "std_z": np.std(f_sim_z),
                "rms_x": u_rms_x,
                "rms_y": u_rms_y,
                "rms_z": u_rms_z,
                "peak_z": np.max(np.abs(f_sim_z)),
                "variance_z": np.var(f_sim_z),
                "xz_ratio": u_rms_x / (u_rms_z + 1e-6),
                "fft_mean": np.mean(user_fft),
                "fft_max": np.max(user_fft),
                "fft_std": np.std(user_fft),
            }
        ]
    )

    print(f"\n[Step 4/4] Processing prediction using {best_model_name}...")

    pred_encoded = best_model.predict(user_features)[0]
    prediction = le.inverse_transform([pred_encoded])[0]

    probabilities = best_model.predict_proba(user_features)[0]
    max_prob = max(probabilities) * 100

    print("\n================== PREDICTION OUTPUT ==================")
    print(f"Tracked Location (GPS): {user_lat} , {user_lon}")
    print(f"Vehicle Velocity (Speed): {user_speed} km/h")
    # এখানে এখন 'BAD', 'REGULAR' অথবা 'GOOD' ডাইনামিকালি দেখাবে
    print(f"Detected Road Surface Condition: **{prediction.upper()}**")
    print(f"Model Confidence Level ({best_model_name}): {max_prob:.2f}%")
    print("=======================================================")

    # --- 7. DUAL VISUALIZATION GRAPHICS ENGINE ---
    print("\nDisplaying interactive analysis graphics panel...")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    try:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.set_window_title(
            f"Road Quality Identification Platform - Inference Engine ({best_model_name})"
        )
    except Exception:
        pass

    # Subplot 1: Heatmap Confusion Matrix
    best_model_test_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_model_test_preds)
    target_names = le.inverse_transform(best_model.classes_)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax[0],
        cbar=True,
    )
    ax[0].set_title(
        f"Confusion Matrix ({best_model_name} Metrics)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax[0].set_xlabel("Predicted Road Status")
    ax[0].set_ylabel("Ground Truth")

    # Subplot 2: Multi-domain Cloud Scatter Map
    user_vibration = np.sum(np.abs(f_sim_z))
    scatter = ax[1].scatter(
        df_ml["speed_kmh"],
        df_ml["total_vibration"],
        c=y_encoded,
        cmap="Set2",
        alpha=0.6,
        label="Historical Pavement Logs",
    )

    ax[1].scatter(
        user_speed,
        user_vibration,
        color="red",
        marker="X",
        s=300,
        edgecolors="black",
        label="Simulated Input Window",
        zorder=10,
    )
    ax[1].set_title(
        f"Vehicle Dynamic Log (Current Assessment: {prediction.upper()})",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax[1].set_xlabel("Vehicle Speed (km/h)")
    ax[1].set_ylabel("Total Window Vibration Accumulation")
    ax[1].grid(True, linestyle="--", alpha=0.5)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

except ValueError:
    print("\n[Error] Invalid Input! Numerical variables mandatory.")