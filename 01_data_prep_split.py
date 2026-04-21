import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python 01_data_prep_split.py <Name> <Path_to_Input_Excel> <Output_Directory>"
        )
        sys.exit(1)

    name = sys.argv[1]
    input_file = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)

    print("Loading and preparing data...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        sys.exit(1)

    # Drop empty phantom rows
    if "Receptor" in df.columns:
        df = df.dropna(subset=["Receptor"])
    else:
        df = df.dropna(how="all")

    if "Classification" not in df.columns:
        print("Error: 'Classification' column not found.")
        sys.exit(1)

    # Normalize Classification column
    df["Classification"] = df["Classification"].astype(str).str.strip()

    def normalize_class(x):
        xl = x.lower()
        if xl == "canonical/immunogenic":
            return "canonical/immunogenic"
        if xl == "evading":
            return "Evading"
        if xl == "deviant":
            return "Deviant"
        if xl == "evading/antagonist":
            return "Evading/antagonist"
        return x

    df["Classification"] = df["Classification"].apply(normalize_class)

    # Filter out missing labels
    valid_classes = [
        "canonical/immunogenic",
        "Evading",
        "Deviant",
        "Evading/antagonist",
    ]
    df_filtered = df[df["Classification"].isin(valid_classes)].copy()

    # Calculate feature averages
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

    # Isolate X (features) and y (target)
    X = df_filtered[features]
    y = df_filtered["Classification"]

    # Drop any complex that failed to calculate scores (NaNs)
    valid_indices = X.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    print(f"Total valid samples for ML: {len(X)}")

    # 85/15 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Verify stratification
    print("\nTest Set Class Distribution:")
    print(y_test.value_counts())

    # Save to CSV
    X_train.to_csv(os.path.join(output_dir, f"{name}_X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, f"{name}_X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, f"{name}_y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, f"{name}_y_test.csv"), index=False)

    print(f"\nData successfully split and saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
