#!/usr/bin/env python3
"""
Merge combined output Excel files from ec3, ec2, and ec directories.
Removes duplicate rows based on all columns.
"""

from pathlib import Path

import pandas as pd


def load_combined_output(directory: str) -> pd.DataFrame:
    """
    Load combined output Excel file from a directory.

    Args:
        directory: Directory name (e.g., 'ec3', 'ec2', 'ec')

    Returns:
        DataFrame with the data, or empty DataFrame if file doesn't exist
    """
    excel_file = Path(directory) / f"{directory}_combine_output.xlsx"

    if not excel_file.exists():
        print(f"⚠️  File not found: {excel_file}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(excel_file, engine="openpyxl")
        print(f"✅ Loaded {excel_file}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading {excel_file}: {e}")
        return pd.DataFrame()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.

    Args:
        df: DataFrame to deduplicate

    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df

    original_count = len(df)

    # Remove duplicates based on all columns
    df_deduped = df.drop_duplicates()

    duplicates_removed = original_count - len(df_deduped)

    if duplicates_removed > 0:
        print(f"   🧹 Removed {duplicates_removed} duplicate row(s)")

    return df_deduped


def main():
    """Main function to merge combined outputs."""
    print("=" * 70)
    print(" " * 15 + "Merge Combined Output Files")
    print("=" * 70)
    print()

    # Directories to process
    directories = ["ec3", "ec2", "ec"]

    # Load all combined output files
    all_dataframes = []

    for directory in directories:
        df = load_combined_output(directory)
        if not df.empty:
            all_dataframes.append(df)
        print()

    if not all_dataframes:
        print("❌ Error: No combined output files found!")
        print("   Expected files:")
        for directory in directories:
            print(f"   - {directory}/{directory}_combine_output.xlsx")
        return

    # Combine all DataFrames
    print("=" * 70)
    print("Merging files...")
    print("=" * 70)
    print()

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    print(f"📊 Total rows before deduplication: {len(combined_df)}")
    print()

    # Remove duplicates
    print("Removing duplicate rows...")
    combined_df = remove_duplicates(combined_df)
    print()

    print(f"📊 Total rows after deduplication: {len(combined_df)}")
    print()

    if combined_df.empty:
        print("⚠️  No data to save after merging and deduplication.")
        return

    # Ensure columns are in correct order
    columns_order = [
        "filename",
        "Sr.No",
        "Document No.& Year",
        "Name of Executant(s)",
        "Name of Claimant(s)",
        "Survey No.",
        "Plot No.",
    ]

    # Only include columns that exist in the DataFrame
    available_columns = [col for col in columns_order if col in combined_df.columns]
    combined_df = combined_df[available_columns]

    # Save merged output
    output_file = "merged_combine_output.xlsx"
    combined_df.to_excel(output_file, index=False, engine="openpyxl")

    print("=" * 70)
    print(" " * 25 + "Summary")
    print("=" * 70)
    print()
    print("📁 Files merged:")
    for directory in directories:
        excel_file = Path(directory) / f"{directory}_combine_output.xlsx"
        if excel_file.exists():
            df = pd.read_excel(excel_file, engine="openpyxl")
            print(f"   - {directory}/{directory}_combine_output.xlsx ({len(df)} rows)")
    print()
    print(f"📊 Final merged file: {output_file}")
    print(f"   Total unique rows: {len(combined_df)}")
    print()

    # Show breakdown by filename
    if "filename" in combined_df.columns:
        print("Rows by source file:")
        filename_counts = combined_df["filename"].value_counts()
        for filename, count in filename_counts.items():
            print(f"   - {filename}: {count} rows")
        print()

    print("=" * 70)
    print("✅ Merge complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
