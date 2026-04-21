import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    if len(sys.argv) != 3:
        print("Usage: python 04_final_testing.py <Name> <Data_Directory>")
        sys.exit(1)

    name = sys.argv[1]
    data_dir = sys.argv[2]

    print("Loading datasets...")
    try:
        # Load the 85% Training Data
        X_train = pd.read_csv(os.path.join(data_dir, f"{name}_X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, f"{name}_y_train.csv"))[
            "Classification"
        ]

        # Load the 15% Test Data (The "Vault")
        X_test = pd.read_csv(os.path.join(data_dir, f"{name}_X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_dir, f"{name}_y_test.csv"))[
            "Classification"
        ]
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(
        "Training the final Logistic Regression model on 100% of the training data..."
    )

    # Create the exact winning pipeline from our evaluation step
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    # Train the model
    final_model.fit(X_train, y_train)

    # Save the trained model for future use
    model_path = os.path.join(data_dir, f"{name}_final_logistic_model.joblib")
    joblib.dump(final_model, model_path)
    print(f"Model successfully saved to: {model_path}")

    print("\nPredicting on the unseen 15% Test Set...")
    # Predict the unseen test data
    y_pred = final_model.predict(X_test)

    # Generate Text Report
    report_file = os.path.join(data_dir, f"{name}_final_test_report.txt")

    with open(report_file, "w") as f:
        f.write("====================================================\n")
        f.write("             FINAL MODEL TEST REPORT                \n")
        f.write("====================================================\n\n")

        acc = accuracy_score(y_test, y_pred)
        f.write(f"Overall Accuracy on Unseen Data: {acc:.4f}\n\n")

        f.write("Detailed Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n====================================================\n")

    print(f"Test report saved to: {report_file}")

    # Generate Confusion Matrix Plot
    class_order = ["canonical/immunogenic", "Evading", "Deviant", "Evading/antagonist"]

    # Calculate confusion matrix enforcing the specific class order
    cm = confusion_matrix(y_test, y_pred, labels=class_order)

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white", font_scale=1.2)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_order,
        yticklabels=class_order,
    )

    plt.title("Confusion Matrix (15% Test Set)", fontsize=18)
    plt.ylabel("Actual True Classification", fontsize=14)
    plt.xlabel("Model's Predicted Classification", fontsize=14)
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_file = os.path.join(data_dir, f"{name}_confusion_matrix.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Confusion matrix plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
