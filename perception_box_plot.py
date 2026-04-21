import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python perception_box_plot.py <Input_File.csv> <Output_Directory>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.2)

    # Check if the input is actually a file
    if not os.path.isfile(input_file):
        print(f"Error: The input '{input_file}' is not a valid file.")
        sys.exit(1)

    filename = os.path.basename(input_file)
    base_name = os.path.splitext(filename)[0]

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)

    # Validate that the file has the necessary structure
    if "Perception" not in df.columns:
        print(f"Error: 'Perception' column not found in {filename}.")
        sys.exit(1)

    # Identify the score column (it should be the only other column besides 'Perception')
    score_cols = [col for col in df.columns if col != "Perception"]

    if len(score_cols) != 1:
        print(
            f"Error: Expected exactly 1 score column besides 'Perception', found {len(score_cols)}."
        )
        sys.exit(1)

    score = score_cols[0]

    # Generate the plot
    plt.figure(figsize=(7, 6))

    # ADDED 'order' PARAMETER HERE TO ENFORCE X-AXIS SEQUENCE
    sns.boxplot(
        x="Perception",
        y=score,
        data=df,
        width=0.5,
        palette="Set2",
        order=["Perception", "No perception"],
    )

    plt.title(f"{score.upper()} by Perception Classification", fontsize=20)
    plt.ylabel(f"Average {score}", fontsize=18)
    plt.xlabel("Perception Label", fontsize=18)
    plt.tight_layout()

    # Save figure with the identical base name
    out_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Successfully generated: {base_name}.png in '{output_dir}'")


if __name__ == "__main__":
    main()
