import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


def main():
    if len(sys.argv) != 3:
        print("Usage: python 03_model_evaluation.py <Name> <Data_Directory>")
        sys.exit(1)

    name = sys.argv[1]
    data_dir = sys.argv[2]

    # Load Training Data
    try:
        X_train = pd.read_csv(os.path.join(data_dir, f"{name}_X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, f"{name}_y_train.csv"))[
            "Classification"
        ]
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)

    # Define the models
    # We use class_weight='balanced' where possible to handle the Evading class imbalance
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42, class_weight="balanced"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        # GradientBoosting doesn't have a direct class_weight parameter in sklearn,
        # so we limit depth to prevent overfitting on the majority class
        "Gradient Boosting": GradientBoostingClassifier(max_depth=3, random_state=42),
    }

    # Setup K-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # We will track the Macro F1-Score for plotting
    cv_f1_scores = []
    cv_model_names = []

    report_file = os.path.join(data_dir, f"{name}_model_comparison_report.txt")

    with open(report_file, "w") as f:
        f.write("====================================================\n")
        f.write("            MODEL CROSS-VALIDATION REPORT           \n")
        f.write("====================================================\n\n")

        print("Evaluating models. This may take a few seconds...\n")

        for model_name, model in models.items():
            # Wrap in a pipeline to scale features within each fold to prevent data leakage
            pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])

            # Run Cross-Validation
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=["accuracy", "balanced_accuracy", "f1_macro"],
            )

            # Extract means and standard deviations
            acc_mean = np.mean(scores["test_accuracy"])
            acc_std = np.std(scores["test_accuracy"])
            bal_acc_mean = np.mean(scores["test_balanced_accuracy"])
            bal_acc_std = np.std(scores["test_balanced_accuracy"])
            f1_mean = np.mean(scores["test_f1_macro"])
            f1_std = np.std(scores["test_f1_macro"])

            # Store F1 scores for the boxplot
            cv_f1_scores.extend(scores["test_f1_macro"])
            cv_model_names.extend([model_name] * 5)

            # Write to report
            f.write(f"--- {model_name} ---\n")
            f.write(f"Accuracy:          {acc_mean:.4f} (+/- {acc_std:.4f})\n")
            f.write(f"Balanced Accuracy: {bal_acc_mean:.4f} (+/- {bal_acc_std:.4f})\n")
            f.write(f"Macro F1-Score:    {f1_mean:.4f} (+/- {f1_std:.4f})\n\n")

            # Print to terminal
            print(
                f"{model_name:20s} | Macro F1: {f1_mean:.4f} | Bal Acc: {bal_acc_mean:.4f}"
            )

        f.write("====================================================\n")
        f.write("Note: Models were evaluated using 5-Fold Stratified CV.\n")
        f.write(
            "Macro F1-Score is the best metric to look at due to class imbalance.\n"
        )
        f.write("====================================================\n")

    print(f"\nEvaluation complete! Report saved to: {report_file}")

    # Generate a Box Plot of the Cross-Validation F1 Scores
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid", font_scale=1.2)

    df_plot = pd.DataFrame({"Model": cv_model_names, "Macro F1-Score": cv_f1_scores})

    sns.boxplot(x="Model", y="Macro F1-Score", data=df_plot, palette="Set2", width=0.5)
    plt.title("Model Comparison (5-Fold Cross Validation)", fontsize=18)
    plt.ylabel("Macro F1-Score", fontsize=16)
    plt.xlabel("Algorithm", fontsize=16)
    plt.xticks(rotation=15)
    plt.tight_layout()

    plot_file = os.path.join(data_dir, f"{name}_model_comparison.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Boxplot saved to: {plot_file}")


if __name__ == "__main__":
    main()
