import os
import csv
import json
import re
import pandas as pd

AF3_OUTPUT_DIR = "/Users/aminoroa/FLS2_Project/af3_output"
OUTPUT_EXCEL   = "/Users/aminoroa/FLS2_Project/input_data/af3_output_data.xlsx"


def extract_value_from_line(line):
    """Strip whitespace, quotes, colons, and trailing commas to get the numeric value."""
    value = line.strip().rstrip(",")
    # handles both ' "iptm": 0.77,' and '  0.36,'
    if ":" in value:
        value = value.split(":")[-1].strip()
    try:
        return float(value)
    except ValueError:
        return None


def get_best_sample(ranking_csv_path):
    """Return {seed: best_sample_number} by highest ranking_score per seed."""
    best = {}
    with open(ranking_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed   = int(row["seed"])
            sample = int(row["sample"])
            score  = float(row["ranking_score"])
            if seed not in best or score > best[seed]["score"]:
                best[seed] = {"sample": sample, "score": score}
    return {seed: v["sample"] for seed, v in best.items()}


def extract_confidence_values(json_path):
    """
    Read specific lines from summary_confidences.json:
      Line 48 -> iptm
      Line 49 -> ptm
      Line 14 -> ligand_fls2_iptm
      Line 16 -> ligand_coreceptor_iptm
      Line 31 -> ligand_pae_min_fls2
      Line 33 -> ligand_pae_min_coreceptor
    Line numbers are 1-based, matching the fixed AF3 output format.
    """
    with open(json_path) as f:
        lines = f.readlines()

    line_map = {
        "iptm":                      48,
        "ptm":                       49,
        "ligand_fls2_iptm":          14,
        "ligand_coreceptor_iptm":    16,
        "ligand_pae_min_fls2":       31,
        "ligand_pae_min_coreceptor": 33,
    }

    result = {}
    for key, lineno in line_map.items():
        if lineno <= len(lines):
            result[key] = extract_value_from_line(lines[lineno - 1])
        else:
            result[key] = None
    return result


# ── Main ────────────────────────────────────────────────────────────────────

pattern = re.compile(r"^atfls2_(.+)_atbak1$")
rows = []

folder_names = sorted(os.listdir(AF3_OUTPUT_DIR))

for folder in folder_names:
    match = pattern.match(folder)
    if not match:
        continue

    peptide_name = match.group(1)
    base_path    = os.path.join(AF3_OUTPUT_DIR, folder, folder)

    ranking_csv  = os.path.join(base_path, "ranking_scores.csv")
    if not os.path.isfile(ranking_csv):
        print(f"  [SKIP] ranking_scores.csv not found for {peptide_name}")
        continue

    best_samples = get_best_sample(ranking_csv)   # {1: sample#, 2: sample#, 3: sample#}

    row = {"peptide_name": peptide_name}

    rep_data = {}
    for seed in [1, 2, 3]:
        sample = best_samples.get(seed)
        if sample is None:
            print(f"  [WARN] seed {seed} not found in ranking_scores for {peptide_name}")
            rep_data[seed] = {k: None for k in
                              ["iptm", "ptm", "ligand_fls2_iptm",
                               "ligand_coreceptor_iptm", "ligand_pae_min_fls2",
                               "ligand_pae_min_coreceptor"]}
            continue

        json_path = os.path.join(
            base_path,
            f"seed-{seed}_sample-{sample}",
            "summary_confidences.json"
        )

        if not os.path.isfile(json_path):
            print(f"  [WARN] {json_path} not found")
            rep_data[seed] = {k: None for k in
                              ["iptm", "ptm", "ligand_fls2_iptm",
                               "ligand_coreceptor_iptm", "ligand_pae_min_fls2",
                               "ligand_pae_min_coreceptor"]}
        else:
            rep_data[seed] = extract_confidence_values(json_path)

    # Build row in the requested column order
    row["Rep1 iptm"]                       = rep_data[1]["iptm"]
    row["Rep1 ptm"]                        = rep_data[1]["ptm"]
    row["Rep2 iptm"]                       = rep_data[2]["iptm"]
    row["Rep2 ptm"]                        = rep_data[2]["ptm"]
    row["Rep3 iptm"]                       = rep_data[3]["iptm"]
    row["Rep3 ptm"]                        = rep_data[3]["ptm"]
    row["Rep1 ligand_fls2_iptm"]           = rep_data[1]["ligand_fls2_iptm"]
    row["Rep1 ligand_coreceptor_iptm"]     = rep_data[1]["ligand_coreceptor_iptm"]
    row["Rep1 ligand_pae_min_fls2"]        = rep_data[1]["ligand_pae_min_fls2"]
    row["Rep1 ligand_pae_min_coreceptor"]  = rep_data[1]["ligand_pae_min_coreceptor"]
    row["Rep2 ligand_fls2_iptm"]           = rep_data[2]["ligand_fls2_iptm"]
    row["Rep2 ligand_coreceptor_iptm"]     = rep_data[2]["ligand_coreceptor_iptm"]
    row["Rep2 ligand_pae_min_fls2"]        = rep_data[2]["ligand_pae_min_fls2"]
    row["Rep2 ligand_pae_min_coreceptor"]  = rep_data[2]["ligand_pae_min_coreceptor"]
    row["Rep3 ligand_fls2_iptm"]           = rep_data[3]["ligand_fls2_iptm"]
    row["Rep3 ligand_coreceptor_iptm"]     = rep_data[3]["ligand_coreceptor_iptm"]
    row["Rep3 ligand_pae_min_fls2"]        = rep_data[3]["ligand_pae_min_fls2"]
    row["Rep3 ligand_pae_min_coreceptor"]  = rep_data[3]["ligand_pae_min_coreceptor"]

    rows.append(row)
    print(f"  [OK] {peptide_name}")

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_EXCEL, index=False)
print(f"\nDone. {len(rows)} peptides written to {OUTPUT_EXCEL}")
