import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, RFE, RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold


def main():
    if len(sys.argv) != 3:
        print("Usage: python 02_feature_ranking.py <Name> <Data_Directory>")
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

    feature_names = X_train.columns

    # Initialize output text file
    output_file = os.path.join(data_dir, f"{name}_feature_ranking_report.txt")

    with open(output_file, "w") as f:
        f.write("====================================================\n")
        f.write("                FEATURE RANKING REPORT              \n")
        f.write("====================================================\n\n")

        # Base estimator for wrapper methods:
        # Using Random Forest with class_weight='balanced' to handle Evading imbalance
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )

        # ---------------------------------------------------------
        # 1. ANOVA F-test (Univariate Feature Selection)
        # ---------------------------------------------------------
        f_scores, p_values = f_classif(X_train, y_train)
        anova_results = pd.DataFrame({"Feature": feature_names, "F-Score": f_scores})
        anova_results = anova_results.sort_values(by="F-Score", ascending=False)

        f.write("1. ANOVA F-TEST (Univariate Importance)\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Measures the ratio of variance between classes to variance within classes.\n"
        )
        f.write("Higher F-score = better separation.\n\n")
        f.write(anova_results.to_string(index=False) + "\n\n\n")

        # ---------------------------------------------------------
        # 2. Tree-based Feature Importance (Mean Decrease in Impurity)
        # ---------------------------------------------------------
        rf.fit(X_train, y_train)
        tree_importances = pd.DataFrame(
            {"Feature": feature_names, "Importance": rf.feature_importances_}
        )
        tree_importances = tree_importances.sort_values(
            by="Importance", ascending=False
        )

        f.write("2. TREE-BASED FEATURE IMPORTANCE (Random Forest)\n")
        f.write("-" * 40 + "\n")
        f.write(
            "How much each feature contributes to decreasing impurity across all trees.\n\n"
        )
        f.write(tree_importances.to_string(index=False) + "\n\n\n")

        # ---------------------------------------------------------
        # 3. Permutation Importance
        # ---------------------------------------------------------
        perm_imp = permutation_importance(
            rf, X_train, y_train, n_repeats=10, random_state=42
        )
        perm_results = pd.DataFrame(
            {"Feature": feature_names, "Importance": perm_imp.importances_mean}
        )
        perm_results = perm_results.sort_values(by="Importance", ascending=False)

        f.write("3. PERMUTATION IMPORTANCE\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Measures model accuracy drop when a specific feature is randomly shuffled.\n\n"
        )
        f.write(perm_results.to_string(index=False) + "\n\n\n")

        # ---------------------------------------------------------
        # 4. Recursive Feature Elimination (RFE)
        # ---------------------------------------------------------
        # Rank features by iteratively dropping the least important one
        rfe = RFE(estimator=rf, n_features_to_select=1)
        rfe.fit(X_train, y_train)
        rfe_results = pd.DataFrame({"Feature": feature_names, "Rank": rfe.ranking_})
        rfe_results = rfe_results.sort_values(by="Rank", ascending=True)

        f.write("4. RECURSIVE FEATURE ELIMINATION (RFE)\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Iteratively drops the weakest feature. Rank 1 means most important.\n\n"
        )
        f.write(rfe_results.to_string(index=False) + "\n\n\n")

        # ---------------------------------------------------------
        # 5. RFECV (Recursive Feature Elimination with Cross-Validation)
        # ---------------------------------------------------------
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rfecv = RFECV(
            estimator=rf, step=1, cv=cv, scoring="accuracy", min_features_to_select=1
        )
        rfecv.fit(X_train, y_train)

        f.write("5. RFECV (RFE with 5-Fold Cross Validation)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Optimal number of features found: {rfecv.n_features_in_}\n")
        f.write(
            "Cross-validation validated rank of features (Rank 1 = selected in optimal set):\n\n"
        )

        rfecv_results = pd.DataFrame({"Feature": feature_names, "Rank": rfecv.ranking_})
        rfecv_results = rfecv_results.sort_values(by="Rank", ascending=True)
        f.write(rfecv_results.to_string(index=False) + "\n\n")

        f.write("====================================================\n")
        f.write("END OF REPORT\n")
        f.write("====================================================\n")

    print(f"Feature ranking complete! Report saved to: {output_file}")

    # Print a quick summary to terminal
    print("\n--- Quick Summary of Top Features ---")
    print(f"Top by Tree-based: {tree_importances.iloc[0]['Feature']}")
    print(f"Top by Permutation: {perm_results.iloc[0]['Feature']}")
    print(f"Optimal feature count by RFECV: {rfecv.n_features_in_}")


if __name__ == "__main__":
    main()
