import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
from datetime import datetime
import re


def scrape_stock_data():
    """
    Scrapes stock data from İş Yatırım website and returns as DataFrame
    """

    # URL to scrape
    url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx#page-1"

    # Headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    try:
        print("Connecting to İş Yatırım website...")

        # Create a session for better connection handling
        session = requests.Session()
        session.headers.update(headers)

        # Make the request
        response = session.get(url, timeout=30)
        response.raise_for_status()

        print(f"Successfully connected. Status code: {response.status_code}")

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for tables in the page
        tables = soup.find_all('table')

        if not tables:
            print("No tables found on the page. The page structure might have changed.")
            return None

        print(f"Found {len(tables)} table(s) on the page.")

        # Try to find the main data table (usually the largest one)
        main_table = None
        max_rows = 0

        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            if len(rows) > max_rows:
                max_rows = len(rows)
                main_table = table
                table_index = i

        if main_table is None:
            print("Could not identify the main data table.")
            return None

        print(f"Using table {table_index + 1} with {max_rows} rows as the main data table.")

        # Extract table data
        data = []
        headers = []

        rows = main_table.find_all('tr')

        # Extract headers from the first row
        if rows:
            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

            # If no proper headers found, create generic ones
            if not headers or all(h == '' for h in headers):
                # Count columns from first data row
                first_data_row = rows[1] if len(rows) > 1 else rows[0]
                col_count = len(first_data_row.find_all(['td', 'th']))
                headers = [f'Column_{i + 1}' for i in range(col_count)]

        # Extract data rows
        for row in rows[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                # Only add rows that have data
                if any(cell.strip() for cell in row_data):
                    data.append(row_data)

        if not data:
            print("No data rows found in the table.")
            return None

        # Ensure all rows have the same number of columns as headers
        max_cols = len(headers)
        cleaned_data = []

        for row in data:
            # Pad or trim row to match header length
            if len(row) < max_cols:
                row.extend([''] * (max_cols - len(row)))
            elif len(row) > max_cols:
                row = row[:max_cols]
            cleaned_data.append(row)

        # Create DataFrame
        df_ozet = pd.DataFrame(cleaned_data, columns=headers)

        print(f"\nSuccessfully created DataFrame with {len(df_ozet)} rows and {len(df_ozet.columns)} columns.")
        print(f"Columns: {list(df_ozet.columns)}")

        return df_ozet

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the website: {e}")
        return None
    except Exception as e:
        print(f"Error processing the data: {e}")
        return None


def scrape_kap_data():
    """
    Scrapes data from KAP website and returns as DataFrame
    """

    # URL to scrape
    url = "https://kap.org.tr/tr/tumKalemler/kpy41_acc5_fiili_dolasimdaki_pay"

    # Headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    try:
        print("Connecting to KAP website...")

        # Create a session for better connection handling
        session = requests.Session()
        session.headers.update(headers)

        # Make the request
        response = session.get(url, timeout=30)
        response.raise_for_status()

        print(f"Successfully connected to KAP. Status code: {response.status_code}")

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for tables in the page
        tables = soup.find_all('table')

        if not tables:
            print("No tables found on KAP page.")
            return None

        print(f"Found {len(tables)} table(s) on KAP page.")

        # Find the main data table
        main_table = None
        max_rows = 0

        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            if len(rows) > max_rows:
                max_rows = len(rows)
                main_table = table
                table_index = i

        if main_table is None:
            print("Could not identify the main data table on KAP.")
            return None

        print(f"Using table {table_index + 1} with {max_rows} rows as the main data table from KAP.")

        # Extract table data
        data = []
        headers = []

        rows = main_table.find_all('tr')

        # Extract headers from the first row
        if rows:
            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

        # Extract data rows
        for row in rows[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                # Only add rows that have data
                if any(cell.strip() for cell in row_data):
                    data.append(row_data)

        if not data:
            print("No data rows found in KAP table.")
            return None

        # Ensure all rows have the same number of columns as headers
        max_cols = len(headers)
        cleaned_data = []

        for row in data:
            # Pad or trim row to match header length
            if len(row) < max_cols:
                row.extend([''] * (max_cols - len(row)))
            elif len(row) > max_cols:
                row = row[:max_cols]
            cleaned_data.append(row)

        # Create DataFrame
        df_temp = pd.DataFrame(cleaned_data, columns=headers)

        print(f"\nSuccessfully created KAP DataFrame with {len(df_temp)} rows and {len(df_temp.columns)} columns.")
        print(f"KAP Columns: {list(df_temp.columns)}")

        return df_temp

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to KAP website: {e}")
        return None
    except Exception as e:
        print(f"Error processing KAP data: {e}")
        return None


def clean_numeric_value(value):
    """
    Clean and convert string values to numeric, handling Turkish number format
    """
    if pd.isna(value) or value == '' or value is None:
        return np.nan

    # Convert to string if not already
    value = str(value)

    # Remove any non-numeric characters except dots, commas, and minus signs
    # Handle Turkish number format (comma as decimal separator, dot as thousands separator)
    value = re.sub(r'[^\d,.-]', '', value)

    # Handle Turkish format: replace comma with dot for decimal
    # But first check if there are both comma and dot
    if ',' in value and '.' in value:
        # Assume dot is thousands separator and comma is decimal
        value = value.replace('.', '').replace(',', '.')
    elif ',' in value:
        # Assume comma is decimal separator
        value = value.replace(',', '.')

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def update_dataframe_types(df_ozet, df_temp):
    """
    Update df_ozet with data from df_temp and set correct column types
    """
    print("\nUpdating df_ozet with KAP data...")

    # Merge dataframes on 'Kod' and 'Borsa Kodu'
    if 'Kod' in df_ozet.columns and 'Borsa Kodu' in df_temp.columns and 'Fiili Dolaşımdaki Pay Tutarı(TL)' in df_temp.columns:
        # Create a mapping dictionary from df_temp
        kap_mapping = df_temp.set_index('Borsa Kodu')['Fiili Dolaşımdaki Pay Tutarı(TL)'].to_dict()

        # Add the new column to df_ozet
        df_ozet['Fiili Dolaşımdaki Pay Tutarı(TL)'] = df_ozet['Kod'].map(kap_mapping)

        # Convert to integer by rounding up (ceiling)
        df_ozet['Fiili Dolaşımdaki Pay Tutarı(TL)'] = df_ozet['Fiili Dolaşımdaki Pay Tutarı(TL)'].apply(
            lambda x: int(np.ceil(clean_numeric_value(x))) if not pd.isna(clean_numeric_value(x)) else np.nan
        )

        print("Successfully added 'Fiili Dolaşımdaki Pay Tutarı(TL)' column from KAP data.")
    else:
        print("Warning: Could not merge data - required columns not found.")
        print(f"df_ozet columns: {list(df_ozet.columns)}")
        print(f"df_temp columns: {list(df_temp.columns) if df_temp is not None else 'df_temp is None'}")

    # Convert specified columns to integer
    integer_columns = ["Kapanış(TL)", "Piyasa Değeri(mn TL)", "Piyasa Değeri(mn $)", "Sermaye(mn TL)"]

    for col in integer_columns:
        if col in df_ozet.columns:
            df_ozet[col] = df_ozet[col].apply(
                lambda x: int(clean_numeric_value(x)) if not pd.isna(clean_numeric_value(x)) else np.nan
            )
            print(f"Converted '{col}' to integer type.")
        else:
            print(f"Warning: Column '{col}' not found in df_ozet.")

    # Convert Halka Açıklık Oranı (%) to percentage
    percentage_column = "Halka AçıklıkOranı (%)"
    if percentage_column in df_ozet.columns:
        df_ozet[percentage_column] = df_ozet[percentage_column].apply(
            lambda x: clean_numeric_value(x) / 100 if not pd.isna(clean_numeric_value(x)) else np.nan
        )
        print(f"Converted '{percentage_column}' to percentage format.")
    else:
        print(f"Warning: Column '{percentage_column}' not found in df_ozet.")

    return df_ozet


def export_to_excel(df, filename_prefix="df_ozet"):
    """
    Export DataFrame to Excel file with current date in filename
    """
    try:
        # Get current date in YYYY-MM-DD format
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{filename_prefix}_{current_date}.xlsx"

        df.to_excel(filename, index=False)
        print(f"Data successfully exported to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return False


def main():
    """
    Main function to execute the scraping and display results
    """
    # Scrape the stock data from İş Yatırım
    df_ozet = scrape_stock_data()

    if df_ozet is None:
        print("Failed to create df_ozet. Exiting.")
        return

    # Scrape data from KAP
    df_temp = scrape_kap_data()

    if df_temp is None:
        print("Failed to create df_temp from KAP data.")
    else:
        print("\n" + "=" * 50)
        print("FIRST 5 ROWS OF df_temp (KAP DATA):")
        print("=" * 50)
        print(df_temp.head())

    # Update df_ozet with KAP data and set correct types
    df_ozet = update_dataframe_types(df_ozet, df_temp)


    # Export to Excel with date in filename
    export_success = export_to_excel(df_ozet)

    if export_success:
        print("\n✓ Data successfully exported to Excel with date in filename")
    else:
        print("\n✗ Failed to export data to Excel")


if __name__ == "__main__":
    main()