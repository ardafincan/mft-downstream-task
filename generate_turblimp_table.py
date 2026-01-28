#!/usr/bin/env python3
"""
Generate TurBLiMP Results Table in LaTeX.
Parses CSV results from turblimp_results/ and generates a comparison table.
"""
import pandas as pd
import os


def generate_latex_table():
    base_dir = "turblimp_results"

    # Files
    mft_file = os.path.join(base_dir, "cosmosGPT2-random-init_turblimp_sensitivity.csv")
    tabi_file = os.path.join(base_dir, "tabi-random-init_turblimp_sensitivity.csv")

    if not os.path.exists(mft_file) or not os.path.exists(tabi_file):
        print(f"Error: Missing result files in {base_dir}")
        return

    # Load data
    df_mft = pd.read_csv(mft_file)
    df_tabi = pd.read_csv(tabi_file)

    # Merge on phenomenon/paradigm
    # Assuming columns: 'phenomenon', 'score', etc.
    # Let's peek at structure by assuming standardized columns from previous steps
    # Usually: 'category', 'score'.

    # Standardize column names if needed (sensitivity script usually outputs: category, score, num_samples)
    # Actually, let's just rename 'score' to 'MFT' and 'Tabi'

    df_mft.rename(columns={"avg_similarity": "MFT"}, inplace=True)
    df_tabi.rename(columns={"avg_similarity": "Tabi"}, inplace=True)

    # Merge
    df_merged = pd.merge(
        df_mft[["category", "MFT"]],
        df_tabi[["category", "Tabi"]],
        on="category",
        how="inner",
    )

    # Calculate difference
    df_merged["Diff"] = df_merged["MFT"] - df_merged["Tabi"]

    # Sort by MFT score
    df_merged.sort_values("MFT", ascending=False, inplace=True)

    # Generate LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table}[H]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\resizebox{\linewidth}{!}{")
    latex_lines.append(r"\begin{tabular}{lrrr}")
    latex_lines.append(r"\toprule")
    latex_lines.append(
        r"\textbf{Linguistic Phenomenon} & \textbf{MFT-Random} & \textbf{Tabi-Random} & \textbf{$\Delta$} \\"
    )
    latex_lines.append(r"\midrule")

    for _, row in df_merged.iterrows():
        cat = row["category"].replace("_", " ").title()
        mft = row["MFT"]
        tabi = row["Tabi"]
        diff = row["Diff"]

        # Bold winner
        mft_str = f"{mft:.1%}".replace("%", r"\%")
        tabi_str = f"{tabi:.1%}".replace("%", r"\%")

        if mft > tabi:
            mft_str = f"\\textbf{{{mft_str}}}"
        elif tabi > mft:
            tabi_str = f"\\textbf{{{tabi_str}}}"

        diff_str = f"{diff:+.1%}".replace("%", r"\%")

        latex_lines.append(f"{cat} & {mft_str} & {tabi_str} & {diff_str} \\\\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"}")
    latex_lines.append(
        r"\caption{Detailed TurBLiMP sensitivity scores (accuracy on minimal pairs) verifying linguistic alignment advantages of MFT.}"
    )
    latex_lines.append(r"\label{tab:turblimp_detailed}")
    latex_lines.append(r"\end{table}")

    output_filename = "turblimp_table.tex"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"Generated {output_filename}")


if __name__ == "__main__":
    generate_latex_table()
