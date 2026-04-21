import sys
import os
import pandas as pd


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python classification_scores.py <Name> <Path_to_Input_Excel> <Output_Directory>"
        )
        sys.exit(1)

    name = sys.argv[1]
    input_file = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        sys.exit(1)

    # Drops any row where the 'Receptor' column is empty to avoid phantom rows
    if "Receptor" in df.columns:
        df = df.dropna(subset=["Receptor"])
    else:
        df = df.dropna(how="all")

    # Ensure the column exists
    if "Classification" not in df.columns:
        print("Error: 'Classification' column not found in the input Excel file.")
        sys.exit(1)

    # Clean the 'Classification' column and force exact matching for the 4 categories
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

    valid_classes = [
        "canonical/immunogenic",
        "Evading",
        "Deviant",
        "Evading/antagonist",
    ]

    # Calculate counts
    counts = df["Classification"].value_counts()
    missing_count = (df["Classification"] == "-").sum() + df["Classification"].isin(
        ["nan", "None", ""]
    ).sum()

    print("\n" + "=" * 45)
    print("--- Classification Counts ---")
    for c in valid_classes:
        print(f"{c}: {counts.get(c, 0)} complexes")
    print(f"No Labeling (-): {missing_count} complexes")
    print("=" * 45 + "\n")

    # Filter out '-' and unknown labels
    df_filtered = df[df["Classification"].isin(valid_classes)].copy()

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

    avg_score_names = []

    for score_name, possible_cols in score_mappings.items():
        actual_cols = [col for col in possible_cols if col in df_filtered.columns]

        if len(actual_cols) >= 3:
            df_filtered[score_name] = df_filtered[actual_cols[:3]].mean(axis=1)
            avg_score_names.append(score_name)
        else:
            print(
                f"Warning: Missing replicate columns for {score_name}. Found: {actual_cols}"
            )

    print("--- Summary Statistics ---")
    stats = df_filtered.groupby("Classification")[avg_score_names].agg(
        ["mean", "std", "min", "max"]
    )
    print(stats)

    stats_file = os.path.join(
        output_dir, f"{name}_classification_summary_statistics.csv"
    )
    stats.to_csv(stats_file)
    print(f"\nSaved summary statistics to {stats_file}\n")

    # Save individual CSV files for each score
    for score in avg_score_names:
        score_df = df_filtered[["Classification", score]].dropna(subset=[score])
        out_csv = os.path.join(output_dir, f"{name}_classification_{score}.csv")
        score_df.to_csv(out_csv, index=False)

    print(f"Data for each score successfully saved as .csv files in '{output_dir}'.")


if __name__ == "__main__":
    main()
