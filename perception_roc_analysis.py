import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python perception_roc_analysis.py <Name> <Path_to_Input_Excel> <Output_Directory>"
        )
        sys.exit(1)

    name = sys.argv[1]
    input_file = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    print("1. Loading and preparing data...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        sys.exit(1)

    # Clean missing receptors
    if "Receptor" in df.columns:
        df = df.dropna(subset=["Receptor"])
    else:
        df = df.dropna(how="all")

    # Clean Perception column
    if "Perception" not in df.columns:
        print("Error: 'Perception' column not found.")
        sys.exit(1)

    df["Perception"] = df["Perception"].astype(str).str.strip()
    df["Perception"] = df["Perception"].apply(
        lambda x: (
            "No perception"
            if x.lower() == "no perception"
            else ("Perception" if x.lower() == "perception" else x)
        )
    )

    # Filter to only binary Perception classes
    df_filtered = df[df["Perception"].isin(["Perception", "No perception"])].copy()

    # Convert 'Perception' to a binary target: 1 for Perception, 0 for No perception
    df_filtered["Target"] = (df_filtered["Perception"] == "Perception").astype(int)

    # Average the scores
    score_mappings = {
        "iptm": ["Rep1 iptm", "Rep2 iptm", "Rep3 iptim", "Rep3 iptm"],
        "ptm": ["Rep1 ptm", "Rep2 ptm", "Rep3 ptim", "Rep3 ptm"],
        "ligand_fls2_iptm": [
            "Rep1 ligand_fls2_iptm",
            "Rep2 ligand_fls2_iptm",
            "Rep3 ligand_fls2_iptm",
        ],
        "ligand_coreceptor_iptm": [
            "Rep1 ligand_coreceptor_iptm",
            "Rep2 ligand_coreceptor_iptm",
            "Rep3 ligand_coreceptor_iptm",
        ],
        "ligand_pae_min_fls2": [
            "Rep1 ligand_pae_min_fls2",
            "Rep2 ligand_pae_min_fls2",
            "Rep3 ligand_pae_min_fls2",
        ],
        "ligand_pae_min_coreceptor": [
            "Rep1 ligand_pae_min_coreceptor",
            "Rep2 ligand_pae_min_coreceptor",
            "Rep3 ligand_pae_min_coreceptor",
        ],
    }

    features = []
    for score_name, possible_cols in score_mappings.items():
        actual_cols = [col for col in possible_cols if col in df_filtered.columns]
        if len(actual_cols) >= 3:
            df_filtered[score_name] = df_filtered[actual_cols[:3]].mean(axis=1)
            features.append(score_name)

    # Save prepared data
    csv_path = os.path.join(output_dir, f"{name}_perception_roc_data.csv")
    df_filtered[["Perception", "Target"] + features].to_csv(csv_path, index=False)
    print(f"Data prepared and saved to {csv_path}\n")

    print("2. Performing ROC Analysis and plotting thresholds...")

    # Text report to summarize best thresholds
    summary_file = os.path.join(output_dir, f"{name}_roc_summary_report.txt")
    with open(summary_file, "w") as f:
        f.write(f"ROC AND THRESHOLD ANALYSIS ({name})\n")
        f.write("=" * 50 + "\n\n")

        for score in features:
            # Drop missing values for this specific score
            temp_df = df_filtered[["Target", score]].dropna()
            y_true = temp_df["Target"].values
            y_score = temp_df[score].values

            # Determine score direction (e.g., lower PAE is better vs higher IPTM is better)
            mean_pos = y_score[y_true == 1].mean()
            mean_neg = y_score[y_true == 0].mean()

            higher_is_better = mean_pos > mean_neg

            # ----------------------------------------------------
            # 1. ROC CURVE CALCULATION
            # ----------------------------------------------------
            if higher_is_better:
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
            else:
                # If lower is better, invert scores to calculate ROC correctly
                fpr, tpr, roc_thresholds = roc_curve(y_true, -y_score)

            roc_auc = auc(fpr, tpr)

            # Plot ROC Curve
            plt.figure(figsize=(7, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=14)
            plt.ylabel("True Positive Rate", fontsize=14)
            plt.title(f"ROC Curve: {score}", fontsize=16)
            plt.legend(loc="lower right", fontsize=14)
            plt.tight_layout()

            roc_plot_path = os.path.join(output_dir, f"{name}_{score}_ROC.png")
            plt.savefig(roc_plot_path, dpi=300)
            plt.close()

            # ----------------------------------------------------
            # 2. ACCURACY VS THRESHOLD CALCULATION
            # ----------------------------------------------------
            unique_thresholds = np.unique(y_score)
            accuracies = []

            for t in unique_thresholds:
                if higher_is_better:
                    preds = (y_score >= t).astype(int)
                else:
                    preds = (y_score <= t).astype(int)

                acc = accuracy_score(y_true, preds)
                accuracies.append(acc)

            best_idx = np.argmax(accuracies)
            best_threshold = unique_thresholds[best_idx]
            best_acc = accuracies[best_idx]

            # Write to summary
            direction_str = ">=" if higher_is_better else "<="
            f.write(f"--- Score: {score} ---\n")
            f.write(f"AUC: {roc_auc:.4f}\n")
            f.write(f"Best Accuracy: {best_acc:.2%}\n")
            f.write(
                f"Optimal Threshold: If score is {direction_str} {best_threshold:.4f}, predict 'Perception'.\n\n"
            )

            # Plot Accuracy vs Threshold
            plt.figure(figsize=(7, 6))
            plt.plot(unique_thresholds, accuracies, color="seagreen", lw=2)
            plt.axvline(
                x=best_threshold,
                color="red",
                linestyle="--",
                label=f"Best Threshold = {best_threshold:.3f}\nMax Accuracy = {best_acc:.2%}",
            )

            plt.xlabel(f"{score} Threshold", fontsize=14)
            plt.ylabel("Accuracy", fontsize=14)
            plt.title(f"Accuracy vs. Threshold: {score}", fontsize=16)
            plt.legend(loc="best", fontsize=12)
            plt.tight_layout()

            acc_plot_path = os.path.join(
                output_dir, f"{name}_{score}_Accuracy_Threshold.png"
            )
            plt.savefig(acc_plot_path, dpi=300)
            plt.close()

            print(
                f"Completed analysis for {score}: AUC = {roc_auc:.3f}, Best Threshold = {best_threshold:.3f}"
            )

    print(
        f"\nAll plots and the summary report were successfully saved to '{output_dir}'."
    )


if __name__ == "__main__":
    main()
