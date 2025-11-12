"""
Web scraper for Ford Fiesta listings from cars.com.

This script uses Selenium WebDriver with Safari to scrape Ford Fiesta
listings from cars.com, extracting key information like title, price,
mileage, and location. Results are saved to a CSV file.

Author: Jeffery Lane
Date: November 2025
"""

import os
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver


# ========================================
# CHANGE THIS NUMBER FOR EACH RUN (1-7)
PAGE_NUMBER = 7
# ========================================


def scrape_cars_page(page_num):
    """
    Scrape Ford Fiesta listings from a single page on cars.com.

    Storytime! I could not for the life of me get the scraper to crawl
    automatically. Originally, I was trying to scrape craigslist, but
    that was like even more difficult with their infinite scrolling pages.
    After that i switched to trying to scrape used car data off of cars.com,
    but they seemed to be blocking the automated requests, so thats why
    this script involves manually adjusting the page number between runs lol.

    Args:
        page_num (int): The page number to scrape (1-7)

    Returns:
        list: List of dictionaries containing car information
    """
    print(f"=== Scraping Page {page_num} ===\n")

    # Initialize Safari WebDriver
    print("Opening Safari...")
    driver = webdriver.Safari()

    # Build URL with the page number
    base_url = "https://www.cars.com/shopping/results/"
    params = (
        "?makes[]=ford&maximum_distance=all&models[]=ford-fiesta"
        f"&page={page_num}&page_size=100&stock_type=used&zip=80205"
    )
    url = base_url + params

    print(f"Loading: {url}")
    driver.get(url)

    # Wait for page to load
    time.sleep(5)

    # Get HTML and parse with BeautifulSoup
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # Find all vehicle cards
    cars = soup.find_all('div', {'class': 'vehicle-card'})
    print(f"Found {len(cars)} cars on this page")

    # Extract data from each car
    page_cars = []
    for car in cars:
        title = car.find('h2')
        price = car.find('span', {'class': 'primary-price'})
        mileage = car.find('div', {'class': 'mileage'})
        location = car.find('div', {'class': 'miles-from'})

        page_cars.append({
            'title': title.text.strip() if title else None,
            'price': price.text.strip() if price else None,
            'mileage': mileage.text.strip() if mileage else None,
            'location': location.text.strip() if location else None
        })

    # Close browser
    print("\nClosing browser...")
    driver.quit()

    return page_cars


def save_to_csv(cars_data, filename='ford_fiestas_all.csv'):
    """
    Save scraped car data to CSV file.

    If the file exists, data is appended. Otherwise, a new file is created.

    Args:
        cars_data (list): List of dictionaries containing car information
        filename (str): Name of the CSV file to save to

    Returns:
        None
    """
    df = pd.DataFrame(cars_data)

    if os.path.exists(filename):
        # Append without header
        df.to_csv(filename, mode='a', header=False, index=False)
        print(f"✅ Appended {len(df)} cars to {filename}")

        # Show total count
        total_df = pd.read_csv(filename)
        print(f"📊 Total cars in file: {len(total_df)}")
    else:
        # Create new file with header
        df.to_csv(filename, index=False)
        print(f"✅ Created {filename} with {len(df)} cars")


def main():
    """Main function to orchestrate the scraping process."""
    # Scrape the specified page
    cars_data = scrape_cars_page(PAGE_NUMBER)

    # Save results to CSV
    save_to_csv(cars_data)

    # Print completion message
    print(f"\nDone with page {PAGE_NUMBER}!")
    print("\n💡 To scrape next page:")
    print(f"   1. Edit this script and change PAGE_NUMBER to "
          f"{PAGE_NUMBER + 1}")
    print("   2. Run the script again")


if __name__ == "__main__":
    main()
