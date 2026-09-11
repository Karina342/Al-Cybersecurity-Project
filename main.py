import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


FEATURES = [
    "duration",
    "packet_count",
    "bytes_sent",
    "bytes_received",
    "failed_logins",
    "unique_ports",
]

LABELS = {
    0: "Normal",
    1: "Suspicious",
}

def generate_dataset(samples: int = 2000) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate a synthetic network-traffic dataset.

    Features:
        duration       - connection duration in seconds
        packet_count   - number of packets
        bytes_sent     - bytes sent
        bytes_received - bytes received
        failed_logins  - failed login attempts
        unique_ports   - number of destination ports

    Labels:
        0 = Normal traffic
        1 = Suspicious traffic
    """

    data = []
    labels = []

    for _ in range(samples):

        # Normal traffic
        if random.random() < 0.70:
            duration = random.uniform(5, 180)
            packet_count = random.randint(20, 700)
            bytes_sent = random.randint(500, 70000)
            bytes_received = random.randint(500, 150000)
            failed_logins = random.randint(0, 2)
            unique_ports = random.randint(1, 6)

            label = 0

        # Suspicious traffic
        else:
            duration = random.uniform(0.1, 60)
            packet_count = random.randint(250, 6000)
            bytes_sent = random.randint(15000, 600000)
            bytes_received = random.randint(500, 150000)
            failed_logins = random.randint(2, 25)
            unique_ports = random.randint(4, 40)

            label = 1

        data.append([
            duration,
            packet_count,
            bytes_sent,
            bytes_received,
            failed_logins,
            unique_ports,
        ])

        labels.append(label)

    dataframe = pd.DataFrame(data, columns=FEATURES)

    return dataframe, np.array(labels)

def train_model(
    dataframe: pd.DataFrame,
    labels: np.ndarray,
) -> Tuple[RandomForestClassifier, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:

    """
    Split the dataset, train the Random Forest model
    and return the trained model and test data.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions, X_train

def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> None:

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print("\n" + "=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)

    print(f"\nAccuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Suspicious"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions)

    print("Confusion Matrix:")
    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Normal", "Suspicious"],
    )

    display.plot()
    plt.title("Network Intrusion Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\nConfusion matrix saved as: confusion_matrix.png")
def show_feature_importance(
    model: RandomForestClassifier,
) -> None:

    importance = model.feature_importances_

    feature_data = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance,
    })

    feature_data = feature_data.sort_values(
        by="Importance",
        ascending=False,
    )

    print("\n" + "=" * 65)
    print("FEATURE IMPORTANCE")
    print("=" * 65)

    for _, row in feature_data.iterrows():

        percentage = row["Importance"] * 100

        print(
            f"{row['Feature']:<18} "
            f"{percentage:6.2f}%"
        )

    # Visualization
    plt.figure(figsize=(9, 5))

    plt.bar(
        feature_data["Feature"],
        feature_data["Importance"],
    )

    plt.xlabel("Network Traffic Feature")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig("feature_importance.png")
    plt.close()

    print("\nFeature importance chart saved as: feature_importance.png")

def analyze_traffic(
    model: RandomForestClassifier,
    traffic: List[float],
) -> int:

    traffic_dataframe = pd.DataFrame(
        [traffic],
        columns=FEATURES,
    )

    prediction = int(
        model.predict(traffic_dataframe)[0]
    )

    probabilities = model.predict_proba(
        traffic_dataframe
    )[0]

    prediction_probability = (
        probabilities[prediction] * 100
    )

    print("\n" + "-" * 65)
    print("NETWORK TRAFFIC ANALYSIS")
    print("-" * 65)

    print(f"Connection duration : {traffic[0]:.2f} sec")
    print(f"Packet count        : {int(traffic[1])}")
    print(f"Bytes sent          : {int(traffic[2])}")
    print(f"Bytes received      : {int(traffic[3])}")
    print(f"Failed logins       : {int(traffic[4])}")
    print(f"Unique ports        : {int(traffic[5])}")

    print("\n" + "-" * 65)

    if prediction == 1:
        print("RESULT: SUSPICIOUS TRAFFIC")
        print("Potential abnormal network activity detected.")
    else:
        print("RESULT: NORMAL TRAFFIC")
        print("No suspicious pattern detected by the model.")

    print(
        f"Prediction probability: "
        f"{prediction_probability:.2f}%"
    )

    print("-" * 65)

    return prediction

def get_user_traffic() -> List[float] | None:

    print("\nEnter network traffic information.")
    print("Please enter numeric values only.\n")

    try:

        duration = float(
            input("Connection duration (seconds): ")
        )

        packet_count = int(
            input("Number of packets: ")
        )

        bytes_sent = int(
            input("Bytes sent: ")
        )

        bytes_received = int(
            input("Bytes received: ")
        )

        failed_logins = int(
            input("Failed login attempts: ")
        )

        unique_ports = int(
            input("Number of unique ports: ")
        )

        # Basic validation
        if (
            duration < 0
            or packet_count < 0
            or bytes_sent < 0
            or bytes_received < 0
            or failed_logins < 0
            or unique_ports < 0
        ):
            print("\nValues cannot be negative.")
            return None

        return [
            duration,
            packet_count,
            bytes_sent,
            bytes_received,
            failed_logins,
            unique_ports,
        ]

    except ValueError:

        print(
            "\nInvalid input. "
            "Please enter numbers only."
        )

        return None

def suspicious_example() -> List[float]:

    return [
        2.5,
        2500,
        250000,
        30000,
        12,
        20,
    ]
def normal_example() -> List[float]:

    return [
        45.0,
        120,
        8000,
        25000,
        0,
        2,
    ]

def show_menu() -> None:

    print("\n" + "=" * 65)
    print("AI CYBERSECURITY - INTRUSION DETECTION")
    print("=" * 65)

    print("1. Analyze custom network traffic")
    print("2. Test suspicious traffic example")
    print("3. Test normal traffic example")
    print("4. Show feature importance")
    print("5. Exit")


def main() -> None:

    print("\nStarting AI Cybersecurity System...")

    print("\nGenerating synthetic network-traffic dataset...")

    dataframe, labels = generate_dataset(
        samples=2000
    )

    print(
        f"Dataset created: "
        f"{len(dataframe)} network-traffic records"
    )

    print("\nTraining Random Forest model...")

    (
        model,
        X_test,
        y_test,
        predictions,
        X_train,
    ) = train_model(
        dataframe,
        labels,
    )

    print("Model training completed.")

    evaluate_model(
        model,
        X_test,
        y_test,
        predictions,
    )

    show_feature_importance(model)

    while True:

        show_menu()

        choice = input(
            "\nChoose an option: "
        ).strip()

        if choice == "1":

            traffic = get_user_traffic()

            if traffic is not None:
                analyze_traffic(
                    model,
                    traffic,
                )

        elif choice == "2":

            print(
                "\nTesting suspicious traffic example..."
            )

            analyze_traffic(
                model,
                suspicious_example(),
            )

        elif choice == "3":

            print(
                "\nTesting normal traffic example..."
            )

            analyze_traffic(
                model,
                normal_example(),
            )

        elif choice == "4":

            show_feature_importance(model)

        elif choice == "5":

            print("\nSystem stopped.")
            break

        else:

            print(
                "\nInvalid option. "
                "Please choose 1-5."
            )

if __name__ == "__main__":
    main()