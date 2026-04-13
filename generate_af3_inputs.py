"""
generate_af3_inputs.py
----------------------
Generate AlphaFold3 JSON input files for all combinations of
FLS2 receptors, flg22 ligands, and BAK1 co-receptors.

Usage:
    python generate_af3_inputs.py \
        --fls2   GmFLS2a AtFLS2 NbFLS2-1 \
        --flg22  Paeflg22 Atuflg22 Pta \
        --bak1   gmbak1 atbak1 \
        --input_dir  ~/FLS2_Project/Input_data \
        --output_dir ~/FLS2_Project/AF3_inputs

Arguments:
    --fls2        One or more FLS2 sequence names (must match FASTA headers exactly)
    --flg22       One or more flg22 sequence names (must match FASTA headers exactly)
    --bak1        One or more coreceptor sequence names (must match FASTA headers exactly)
    --input_dir   Directory containing FLS2.fasta, flg22.fasta, coreceptor.fasta
    --output_dir  Directory where JSON files will be saved
"""

import os
import sys
import json
import argparse
import itertools


# ---------------------------------------------------------------------------
# FASTA parser
# ---------------------------------------------------------------------------
def parse_fasta(filepath: str) -> dict[str, str]:
    """Parse a FASTA file and return {header: sequence} dict."""
    sequences = {}
    current_name = None
    current_seq = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    sequences[current_name] = "".join(current_seq)
                current_name = line[1:].strip()  # strip the '>'
                current_seq = []
            else:
                current_seq.append(line)

    if current_name is not None:
        sequences[current_name] = "".join(current_seq)

    return sequences


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------
def build_af3_json(name: str, fls2_seq: str, flg22_seq: str, bak1_seq: str) -> dict:
    """Build the AlphaFold3 input dict for one combination."""
    return {
        "name": name,
        "sequences": [
            {"protein": {"id": "A", "sequence": fls2_seq}},
            {"protein": {"id": "B", "sequence": flg22_seq}},
            {"protein": {"id": "C", "sequence": bak1_seq}},
        ],
        "modelseeds": [1, 2, 3],
        "dialect": "alphafold3",
        "version": 1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate AlphaFold3 JSON inputs for FLS2-flg22-BAK1 complexes."
    )
    parser.add_argument(
        "--fls2",
        nargs="+",
        required=True,
        help="FLS2 sequence name(s) as in FASTA headers",
    )
    parser.add_argument(
        "--flg22",
        nargs="+",
        required=True,
        help="flg22 sequence name(s) as in FASTA headers",
    )
    parser.add_argument(
        "--bak1",
        nargs="+",
        required=True,
        help="Coreceptor sequence name(s) as in FASTA headers",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing FLS2.fasta, flg22.fasta, coreceptor.fasta",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory where JSON files will be saved"
    )
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)

    # ------------------------------------------------------------------
    # Load FASTA files
    # ------------------------------------------------------------------
    fasta_files = {
        "fls2": os.path.join(input_dir, "FLS2.fasta"),
        "flg22": os.path.join(input_dir, "flg22.fasta"),
        "coreceptor": os.path.join(input_dir, "coreceptor.fasta"),
    }

    databases = {}
    for key, path in fasta_files.items():
        if not os.path.isfile(path):
            print(f"[ERROR] FASTA file not found: {path}")
            sys.exit(1)
        databases[key] = parse_fasta(path)
        print(
            f"[INFO]  Loaded {len(databases[key])} sequences from {os.path.basename(path)}"
        )

    # ------------------------------------------------------------------
    # Validate requested names against loaded sequences
    # ------------------------------------------------------------------
    def find_sequence(name: str, db: dict, label: str) -> str:
        """Look up a name; strip surrounding whitespace from headers."""
        # Try exact match first
        if name in db:
            return db[name]
        # Try stripping trailing whitespace/tabs from stored keys
        for key, seq in db.items():
            if key.strip() == name.strip():
                return seq
        available = "\n    ".join(db.keys())
        print(f"[ERROR] '{name}' not found in {label}.\n  Available:\n    {available}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate all combinations
    # ------------------------------------------------------------------
    combos = list(itertools.product(args.fls2, args.flg22, args.bak1))
    print(f"[INFO]  Generating {len(combos)} JSON file(s) → {output_dir}\n")

    for fls2_name, flg22_name, bak1_name in combos:
        fls2_seq = find_sequence(fls2_name, databases["fls2"], "FLS2.fasta")
        flg22_seq = find_sequence(flg22_name, databases["flg22"], "flg22.fasta")
        bak1_seq = find_sequence(bak1_name, databases["coreceptor"], "coreceptor.fasta")

        # Build lowercase filename:  fls2name_flg22name_bak1name.json
        safe = lambda s: s.lower().replace(" ", "_").replace("/", "-")
        file_stem = f"{safe(fls2_name)}_{safe(flg22_name)}_{safe(bak1_name)}"
        json_name = file_stem  # used as "name" field inside JSON too
        out_path = os.path.join(output_dir, file_stem + ".json")

        af3_dict = build_af3_json(json_name, fls2_seq, flg22_seq, bak1_seq)

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(af3_dict, fh, indent=4)

        print(f"  [OK]  {file_stem}.json")

    print(f"\nDone. {len(combos)} file(s) saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
