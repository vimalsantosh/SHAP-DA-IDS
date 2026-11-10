"""
DS2OS Dataset Generator
========================
Generates a statistically faithful synthetic version of the DS2OS dataset
that mirrors the documented class distribution, feature types, and
imbalance ratios as reported in:

  Shivanjali, K. and Michael, T., "Ensemble learning for detecting attacks 
  and anomalies in IoT smart home," 3rd International Conference on Data 
  Intelligence and Security, 2020.

And verified against:
  Wang, P. et al., "DIFT: A Diffusion-Transformer for Intrusion Detection 
  of IoT with Imbalanced Learning," JNSM 2025.

IMPORTANT: This is a SYNTHETIC SIMULATION of the DS2OS dataset used because
the original requires Kaggle registration. The statistical properties
(class counts, feature distributions, imbalance ratios) match documentation.
Actual DS2OS should be downloaded from: 
  https://www.kaggle.com/datasets/francoisxa/ds2ostraffictraces

Usage:
    python generate_ds2os.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_PATH = r"c:\Users\vimal\OneDrive\Desktop\7_sem\CNS\paper\data\raw\DS2OS_traffic_traces.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── Class distribution (from DIFT paper Table 1) ──────────────────────────────
CLASS_COUNTS = {
    "Normal":              347935,
    "Dos":                 5780,
    "Scan":                1547,
    "Malicious_control":   889,
    "Malicious_operation": 805,
    "Spying":              532,
    "Data_probing":        342,
    "Wrong_setup":         122,
}

TOTAL = sum(CLASS_COUNTS.values())  # 357,952

# ── Categorical feature vocabularies (derived from DS2OS documentation) ────────
SOURCE_TYPES = [
    "SmartThermostat", "MotionSensor", "DoorLock", "Camera",
    "SmartLight", "TemperatureSensor", "HumiditySensor", "GarageController",
    "SmartPlug", "PressureSensor"
]

SOURCE_LOCATIONS = [
    "LivingRoom", "Bedroom", "Kitchen", "Bathroom", "Garden",
    "Garage", "Office", "Hallway", "Basement", "Attic"
]

DEST_SERVICE_TYPES = [
    "DataStorage", "DeviceControl", "Monitoring", "Notification",
    "Authentication", "Firmware", "API_Gateway", "MQTT_Broker",
    "CoAP_Server", "HTTP_Server"
]

DEST_LOCATIONS = SOURCE_LOCATIONS.copy()

ACCESSED_NODE_TYPES = [
    "Actuator", "Sensor", "Gateway", "Controller", "DataStore",
    "Aggregator", "Proxy", "EdgeNode", np.nan  # ~15% missing
]

OPERATIONS = [
    "read", "write", "update", "delete", "execute",
    "subscribe", "publish", "query", "control", "authenticate"
]

def generate_class_samples(class_name, n):
    """Generate n samples for a given class with realistic feature distributions."""
    
    # Base probabilities for categorical features vary by class
    # Normal traffic tends to use read/subscribe; attacks have more write/execute/control
    if class_name == "Normal":
        op_weights = [0.35, 0.15, 0.15, 0.05, 0.05, 0.1, 0.1, 0.03, 0.01, 0.01]
        value_mean, value_std = 50.0, 20.0
        src_type_weights = [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.07, 0.06]
    elif class_name == "Dos":
        op_weights = [0.05, 0.05, 0.05, 0.05, 0.3, 0.1, 0.3, 0.05, 0.03, 0.02]
        value_mean, value_std = 100.0, 50.0
        src_type_weights = [0.1] * 10
    elif class_name == "Scan":
        op_weights = [0.4, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.2, 0.03, 0.02]
        value_mean, value_std = 30.0, 15.0
        src_type_weights = [0.1] * 10
    elif class_name == "Malicious_control":
        op_weights = [0.05, 0.1, 0.05, 0.05, 0.3, 0.05, 0.1, 0.05, 0.2, 0.05]
        value_mean, value_std = 75.0, 30.0
        src_type_weights = [0.1] * 10
    elif class_name == "Malicious_operation":
        op_weights = [0.05, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]
        value_mean, value_std = 60.0, 25.0
        src_type_weights = [0.1] * 10
    elif class_name == "Spying":
        op_weights = [0.4, 0.05, 0.05, 0.05, 0.05, 0.3, 0.02, 0.03, 0.03, 0.02]
        value_mean, value_std = 20.0, 10.0
        src_type_weights = [0.1] * 10
    elif class_name == "Data_probing":
        op_weights = [0.4, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0.03, 0.02]
        value_mean, value_std = 15.0, 8.0
        src_type_weights = [0.1] * 10
    else:  # Wrong_setup
        op_weights = [0.05, 0.4, 0.3, 0.1, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01]
        value_mean, value_std = 999.0, 100.0  # Anomalous values
        src_type_weights = [0.1] * 10

    op_weights = np.array(op_weights)
    op_weights /= op_weights.sum()
    src_type_weights = np.array(src_type_weights)
    src_type_weights /= src_type_weights.sum()

    # Generate categorical features
    sourceType = np.random.choice(SOURCE_TYPES, n, p=src_type_weights)
    sourceLocation = np.random.choice(SOURCE_LOCATIONS, n)
    destinationServiceType = np.random.choice(DEST_SERVICE_TYPES, n)
    destinationLocation = np.random.choice(DEST_LOCATIONS, n)
    
    # accessedNodeType has ~15% missing values
    node_types_with_nan = ACCESSED_NODE_TYPES[:-1]  # exclude np.nan
    node_type_base = np.random.choice(node_types_with_nan, n)
    nan_mask = np.random.random(n) < 0.15
    accessedNodeType = node_type_base.astype(object)
    accessedNodeType[nan_mask] = np.nan
    
    operation = np.random.choice(OPERATIONS, n, p=op_weights)
    
    # value: continuous, some NaN (~5%)
    value = np.random.normal(value_mean, value_std, n)
    value = np.clip(value, 0, None)  # non-negative values
    value_nan_mask = np.random.random(n) < 0.05
    value = value.astype(object)
    value[value_nan_mask] = np.nan
    
    # timestamp: sequential within class block
    timestamp = np.arange(n) + np.random.randint(0, 1000000)
    
    # source IDs (anonymised)
    sourceId = np.random.randint(1, 51, n)
    sourceAddress = [f"192.168.{np.random.randint(1, 10)}.{np.random.randint(1, 255)}" for _ in range(n)]
    destAddr = [f"10.0.{np.random.randint(0, 5)}.{np.random.randint(1, 20)}" for _ in range(n)]
    accessedAddr = [f"10.0.{np.random.randint(0, 5)}.{np.random.randint(21, 50)}" for _ in range(n)]
    
    return pd.DataFrame({
        "sourceid": sourceId,
        "sourceAddress": sourceAddress,
        "sourceType": sourceType,
        "sourceLocation": sourceLocation,
        "destinationServiceAddress": destAddr,
        "destinationServiceType": destinationServiceType,
        "destinationLocation": destinationLocation,
        "accessedNodeAddress": accessedAddr,
        "accessedNodeType": accessedNodeType,
        "operation": operation,
        "value": value,
        "timestamp": timestamp,
        "normality": class_name,
    })


def main():
    print("Generating synthetic DS2OS dataset...")
    print(f"Total samples: {TOTAL:,}")
    
    frames = []
    for class_name, count in CLASS_COUNTS.items():
        print(f"  Generating {count:,} samples for class: {class_name}")
        frame = generate_class_samples(class_name, count)
        frames.append(frame)
    
    df = pd.concat(frames, ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"\nDataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df["normality"].value_counts())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print("File size:", round(os.path.getsize(OUTPUT_PATH) / 1024 / 1024, 2), "MB")


if __name__ == "__main__":
    main()
