# Al-Cybersecurity-Project
This project applies Artificial Intelligence and Machine Learning to cybersecurity. It analyzes network traffic to detect suspicious activity and cyberattacks. A dataset is processed, and a machine learning model learns to classify normal and malicious traffic. Project demonstrates how AI can improve automated threat detection and network security.
 """
AI + Cybersecurity
Network Intrusion Detection using Machine Learning

Educational project:
- Generates a small synthetic network-traffic dataset
- Trains a Random Forest classifier
- Evaluates the model
- Allows the user to test new network traffic

Run:
    python main.py
"""

import random
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def generate_dataset(samples: int = 1000):
    """
    Create synthetic network-traffic data.

    Features:
        duration       - connection duration in seconds
        packet_count   - number of packets
        bytes_sent     - bytes sent
        bytes_received - bytes received
        failed_logins  - number of failed login attempts
        unique_ports   - number of different destination ports
    """

    data: List[List[float]] = []
    labels: List[int] = []

    for _ in range(samples):
        # Normal traffic
        if random.random() < 0.70:
            duration = random.uniform(1, 120)
            packet_count = random.randint(10, 500)
            bytes_sent = random.randint(500, 50000)
            bytes_received = random.randint(500, 100000)
            failed_logins = random.randint(0, 2)
            unique_ports = random.randint(1, 5)

            label = 0

        # Suspicious traffic
        else:
            duration = random.uniform(0.1, 30)
            packet_count = random.randint(300, 5000)
            bytes_sent = random.randint(20000, 500000)
            bytes_received = random.randint(1000, 100000)
            failed_logins = random.randint(3, 20)
            unique_ports = random.randint(5, 30)

            label = 1

        data.append([
            duration,
            packet_count,
            bytes_sent,
            bytes_received,
            failed_logins,
            unique_ports
        ])

        labels.append(label)

    return np.array(data), np.array(labels)

def train_model():
    """Generate data, train the classifier and evaluate it."""

    X, y = generate_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 55)
    print("AI CYBERSECURITY - INTRUSION DETECTION")
    print("=" * 55)

    print(f"\nModel accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Suspicious"]
        )
    )

    return model



def analyze_traffic(model, traffic):
    """
    Analyze one network-traffic record.

    traffic:
        [duration, packet_count, bytes_sent,
         bytes_received, failed_logins, unique_ports]
    """

    prediction = model.predict([traffic])[0]
    probabilities = model.predict_proba([traffic])[0]

    confidence = probabilities[prediction] * 100

    print("\n" + "-" * 55)
    print("TRAFFIC ANALYSIS")
    print("-" * 55)

    print(f"Connection duration: {traffic[0]:.2f} seconds")
    print(f"Packet count:        {traffic[1]}")
    print(f"Bytes sent:          {traffic[2]}")
    print(f"Bytes received:      {traffic[3]}")
    print(f"Failed logins:       {traffic[4]}")
    print(f"Unique ports:        {traffic[5]}")

    if prediction == 1:
        print("\n RESULT: SUSPICIOUS TRAFFIC")
    else:
        print("\n RESULT: NORMAL TRAFFIC")

    print(f"Model confidence: {confidence:.2f}%")

    return prediction


def get_user_traffic():
    """Read network-traffic information from the terminal."""

    print("\nEnter network traffic information.")
    print("Use numbers only.\n")

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

        return [
            duration,
            packet_count,
            bytes_sent,
            bytes_received,
            failed_logins,
            unique_ports
        ]

    except ValueError:
        print("\n Invalid input. Please enter numbers only.")
        return 

def main():
    """Run the cybersecurity detection system."""

    print("Starting AI cybersecurity system...")

    model = train_model()

    while True:
        print("\n" + "=" * 55)
        print("MENU")
        print("=" * 55)
        print("1. Analyze network traffic")
        print("2. Test suspicious traffic example")
        print("3. Test normal traffic example")
        print("4. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            traffic = get_user_traffic()

            if traffic is not None:
                analyze_traffic(model, traffic)

        elif choice == "2":
            suspicious_example = [
                2.5,       # duration
                2500,      # packets
                250000,    # bytes sent
                30000,     # bytes received
                12,        # failed logins
                20         # unique ports
            ]

            analyze_traffic(model, suspicious_example)

        elif choice == "3":
            normal_example = [
                45.0,      # duration
                120,       # packets
                8000,      # bytes sent
                25000,     # bytes received
                0,         # failed logins
                2          # unique ports
            ]

            analyze_traffic(model, normal_example)

        elif choice == "4":
            print("\nSystem stopped.")
            break

        else:
            print("   ")

if __name__ == "__main__":
    main()