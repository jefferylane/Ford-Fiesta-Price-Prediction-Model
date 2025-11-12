"""
Data cleaning script for Ford Fiesta listings.

This script removes listings that are not priced from the raw scraped data,
producing a cleaned dataset ready for further processing.

Author: Jeffery Lane
Date: November 2025
"""

import pandas as pd


def clean_ford_fiesta_data(input_file, output_file):
    """
    Clean Ford Fiesta data by removing unpriceable listings.

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the cleaned CSV file

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Load the raw scraped data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Original dataset size: {len(df)} rows")

    # Filter out rows where price is 'Not Priced'
    df_clean = df[df['price'] != 'Not Priced']

    print(f"Cleaned dataset size: {len(df_clean)} rows")
    print(f"Removed {len(df) - len(df_clean)} unpriceable listings")

    # Save cleaned data
    df_clean.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to {output_file}")

    return df_clean


def main():
    """Main function to execute the cleaning process."""
    input_file = 'ford_fiestas_all.csv'
    output_file = 'ford_fiestas_clean.csv'

    # Clean the data
    df_clean = clean_ford_fiesta_data(input_file, output_file)

    # Display summary statistics
    print("\n📊 Summary of cleaned data:")
    print(df_clean.info())


if __name__ == "__main__":
    main()
