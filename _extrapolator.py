"""
Feature extraction script for Ford Fiesta listings.

This script processes cleaned Ford Fiesta data to extract structured features
including year, trim level, numeric price and mileage, state, and distance.

Author: Jeffery Lane
Date: November 2025
"""

import pandas as pd


def extract_year(title_series):
    """
    Extract year from vehicle title.

    Args:
        title_series (pd.Series): Series containing vehicle titles

    Returns:
        pd.Series: Series of years as integers
    """
    # Take first 4 characters (the year)
    year = title_series.str[:4].astype(int)
    return year


def extract_trim(title_series):
    """
    Extract trim level from vehicle title.

    Args:
        title_series (pd.Series): Series containing vehicle titles

    Returns:
        pd.Series: Series of trim levels
    """
    # Split on "Ford Fiesta " and take the part after it
    trim = title_series.str.split('Ford Fiesta ').str[1]
    return trim


def clean_price(price_series):
    """
    Convert price strings to integers.

    Args:
        price_series (pd.Series): Series containing price strings

    Returns:
        pd.Series: Series of prices as integers
    """
    # Remove $, remove commas, then convert to integer
    price = (price_series
             .str.replace('$', '', regex=False)
             .str.replace(',', '', regex=False)
             .astype(int))
    return price


def clean_mileage(mileage_series):
    """
    Convert mileage strings to integers.

    Args:
        mileage_series (pd.Series): Series containing mileage strings

    Returns:
        pd.Series: Series of mileage values as integers
    """
    # Remove commas, remove " mi." unit, then convert to integer
    mileage = (mileage_series
               .str.replace(',', '', regex=False)
               .str.replace(' mi.', '', regex=False)
               .astype(int))
    return mileage


def extract_state(location_series):
    """
    Extract state abbreviation from location string.

    Args:
        location_series (pd.Series): Series containing location strings

    Returns:
        pd.Series: Series of state abbreviations
    """
    # Extract state (2 letters after the comma)
    state = location_series.str.split(', ').str[1].str[:2]
    return state


def extract_distance(location_series):
    """
    Extract distance in miles from location string.

    Args:
        location_series (pd.Series): Series containing location strings

    Returns:
        pd.Series: Series of distances as integers
    """
    # Extract distance using regex pattern
    distance = location_series.str.extract(r'\(([0-9,]+) mi\.\)')[0]
    distance = distance.str.replace(',', '', regex=False).astype(int)
    return distance


def process_ford_fiesta_data(input_file, output_file):
    """
    Process Ford Fiesta data to extract all features.

    Args:
        input_file (str): Path to the cleaned input CSV file
        output_file (str): Path to save the processed CSV file

    Returns:
        pd.DataFrame: Processed dataframe with extracted features
    """
    # Load cleaned data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Processing {len(df)} rows...")

    # Extract features
    df['year'] = extract_year(df['title'])
    df['trim'] = extract_trim(df['title'])
    df['price'] = clean_price(df['price'])
    df['mileage'] = clean_mileage(df['mileage'])
    df['state'] = extract_state(df['location'])
    df['distance'] = extract_distance(df['location'])

    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"✅ Processed data saved to {output_file}")

    return df


def main():
    """Main function to execute the feature extraction process."""
    input_file = 'ford_fiestas_clean.csv'
    output_file = 'ford_fiestas_extrap.csv'

    # Process the data
    df = process_ford_fiesta_data(input_file, output_file)

    # Display summary
    print("\n📊 Processed data summary:")
    print(df.head(10))
    print("\n" + "="*50)
    print(df.info())


if __name__ == "__main__":
    main()
