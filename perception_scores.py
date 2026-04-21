import sys
import os
import pandas as pd


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python perception_scores.py <Name> <Path_to_Input_Excel> <Output_Directory>"
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

    # Clean the 'Perception' column and force exact matching for "No perception"
    df["Perception"] = df["Perception"].astype(str).str.strip()
    df["Perception"] = df["Perception"].apply(
        lambda x: (
            "No perception"
            if x.lower() == "no perception"
            else ("Perception" if x.lower() == "perception" else x)
        )
    )

    perc_count = (df["Perception"] == "Perception").sum()
    no_perc_count = (df["Perception"] == "No perception").sum()
    missing_count = (df["Perception"] == "-").sum() + df["Perception"].isin(
        ["nan", "None", ""]
    ).sum()

    print("\n" + "=" * 45)
    print("--- Perception Classification Counts ---")
    print(f"Perception: {perc_count} complexes")
    print(f"No perception: {no_perc_count} complexes")
    print(f"No Labeling (-): {missing_count} complexes")
    print("=" * 45 + "\n")

    df_filtered = df[df["Perception"].isin(["Perception", "No perception"])].copy()

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
    stats = df_filtered.groupby("Perception")[avg_score_names].agg(
        ["mean", "std", "min", "max"]
    )
    print(stats)

    stats_file = os.path.join(output_dir, f"{name}_summary_statistics.csv")
    stats.to_csv(stats_file)
    print(f"\nSaved summary statistics to {stats_file}\n")

    # Save individual CSV files for each score instead of plotting
    for score in avg_score_names:
        score_df = df_filtered[["Perception", score]].dropna(subset=[score])
        out_csv = os.path.join(output_dir, f"{name}_{score}.csv")
        score_df.to_csv(out_csv, index=False)

    print(f"Data for each score successfully saved as .csv files in '{output_dir}'.")


if __name__ == "__main__":
    main()
