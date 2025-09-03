import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime, timedelta


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


def scrape_circulation_data():
    """
    Scrapes circulation data from KAP website and returns as DataFrame
    """

    # URL for circulation data
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
        print("\nConnecting to KAP website for circulation data...")

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

        # Try to find the main data table
        main_table = None
        max_rows = 0

        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            if len(rows) > max_rows:
                max_rows = len(rows)
                main_table = table
                table_index = i

        if main_table is None:
            print("Could not identify the main data table on KAP page.")
            return None

        print(f"Using table {table_index + 1} with {max_rows} rows as the main circulation data table.")

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
            print("No data rows found in the circulation table.")
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
        df_dolasim = pd.DataFrame(cleaned_data, columns=headers)

        print(
            f"Successfully created circulation DataFrame with {len(df_dolasim)} rows and {len(df_dolasim.columns)} columns.")
        print(f"Circulation columns: {list(df_dolasim.columns)}")

        return df_dolasim

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to KAP website: {e}")
        return None
    except Exception as e:
        print(f"Error processing circulation data: {e}")
        return None


def merge_dataframes(df_ozet, df_dolasim):
    """
    Merges df_ozet with df_dolasim based on stock codes
    """

    if df_ozet is None or df_dolasim is None:
        print("One or both DataFrames are None. Cannot perform merge.")
        return df_ozet

    print("\nStarting DataFrame merge process...")

    # Check if required columns exist
    kod_col = None
    borsa_kodu_col = None

    # Find 'Kod' column in df_ozet (case insensitive)
    for col in df_ozet.columns:
        if 'kod' in col.lower():
            kod_col = col
            break

    # Find 'Borsa Kodu' column in df_dolasim (case insensitive)
    for col in df_dolasim.columns:
        if 'borsa' in col.lower() and 'kod' in col.lower():
            borsa_kodu_col = col
            break

    if kod_col is None:
        print("Warning: 'Kod' column not found in df_ozet. Looking for alternative code columns...")
        # Look for any column that might contain stock codes
        for col in df_ozet.columns:
            if any(keyword in col.lower() for keyword in ['symbol', 'ticker', 'hisse']):
                kod_col = col
                break

    if borsa_kodu_col is None:
        print("Warning: 'Borsa Kodu' column not found in df_dolasim. Looking for alternative code columns...")
        # Look for any column that might contain stock codes
        for col in df_dolasim.columns:
            if any(keyword in col.lower() for keyword in ['symbol', 'ticker', 'kod', 'hisse']):
                borsa_kodu_col = col
                break

    if kod_col is None or borsa_kodu_col is None:
        print(f"Cannot find matching columns. df_ozet columns: {list(df_ozet.columns)}")
        print(f"df_dolasim columns: {list(df_dolasim.columns)}")
        return df_ozet

    print(f"Matching '{kod_col}' from df_ozet with '{borsa_kodu_col}' from df_dolasim")

    # Clean the code columns for better matching
    df_ozet_clean = df_ozet.copy()
    df_dolasim_clean = df_dolasim.copy()

    df_ozet_clean[kod_col] = df_ozet_clean[kod_col].astype(str).str.strip().str.upper()
    df_dolasim_clean[borsa_kodu_col] = df_dolasim_clean[borsa_kodu_col].astype(str).str.strip().str.upper()

    # Handle one-to-many matches by keeping only the first occurrence
    df_dolasim_dedup = df_dolasim_clean.drop_duplicates(subset=[borsa_kodu_col], keep='first')

    if len(df_dolasim_clean) > len(df_dolasim_dedup):
        duplicates_removed = len(df_dolasim_clean) - len(df_dolasim_dedup)
        print(f"Removed {duplicates_removed} duplicate entries from circulation data (keeping first occurrence)")

    # Perform left join to expand df_ozet
    merged_df = df_ozet_clean.merge(
        df_dolasim_dedup,
        left_on=kod_col,
        right_on=borsa_kodu_col,
        how='left'
    )

    # Remove the duplicate borsa_kodu_col if it was added
    if borsa_kodu_col in merged_df.columns and borsa_kodu_col != kod_col:
        merged_df = merged_df.drop(columns=[borsa_kodu_col])

    matches = merged_df[merged_df.iloc[:, len(df_ozet.columns):].notna().any(axis=1)]
    no_matches = len(df_ozet) - len(matches)

    print(f"Merge completed:")
    print(f"- Total rows in df_ozet: {len(df_ozet)}")
    print(f"- Rows with matches: {len(matches)}")
    print(f"- Rows with no matches (filled with NULL): {no_matches}")

    return merged_df


def get_last_workday():
    """
    Returns the current date if it's a workday, otherwise returns the last workday
    """

    current_date = datetime.now()

    # Monday = 0, Tuesday = 1, ..., Sunday = 6
    # Saturday = 5, Sunday = 6 are weekends

    if current_date.weekday() < 5:  # Monday to Friday (0-4)
        workday = current_date
        print(f"Current date {current_date.strftime('%Y-%m-%d')} is a workday.")
    else:
        # Find the last Friday
        days_since_friday = current_date.weekday() - 4  # Friday is 4
        if days_since_friday <= 0:
            days_since_friday += 7

        workday = current_date - timedelta(days=days_since_friday)
        print(f"Current date {current_date.strftime('%Y-%m-%d')} is a weekend.")
        print(f"Using last workday: {workday.strftime('%Y-%m-%d')}")

    return workday


def add_date_column(df_ozet):
    """
    Adds current workday date as a new column to df_ozet
    """

    if df_ozet is None or df_ozet.empty:
        print("Cannot add date column: DataFrame is None or empty")
        return df_ozet

    # Get the appropriate workday date
    workday = get_last_workday()
    date_str = workday.strftime('%Y-%m-%d')

    # Add the date column
    df_ozet['Tarih'] = date_str

    print(f"\nAdded 'Tarih' column with date: {date_str}")
    print(f"Date added to all {len(df_ozet)} rows in df_ozet")

    return df_ozet


def export_to_excel(df_ozet, base_filename="df_ozet"):
    """
    Exports the DataFrame to an Excel file with current date in filename
    """

    if df_ozet is None or df_ozet.empty:
        print("Cannot export: DataFrame is None or empty")
        return False

    try:
        # Get the current workday for filename
        workday = get_last_workday()
        date_str = workday.strftime('%Y%m%d')  # Format: YYYYMMDD for filename

        # Create filename with date
        filename = f"{base_filename}_{date_str}.xlsx"

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        print(f"\nExporting DataFrame to Excel...")
        print(f"File name: {filename}")
        print(f"File path: {file_path}")

        # Export to Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_ozet.to_excel(writer, sheet_name='Stock_Data', index=False)

        print(f"Successfully exported {len(df_ozet)} rows to {filename}")
        print(f"File saved in: {script_dir}")

        return True

    except ImportError:
        print("Error: openpyxl library is required for Excel export.")
        print("Please install it using: pip install openpyxl")
        return False
    except PermissionError:
        print(f"Error: Permission denied. Cannot write to {file_path}")
        print("Make sure the file is not open in Excel or another application.")
        return False
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return False


def main():
    """
    Main function to execute the scraping and display results
    """
    # Scrape the first dataset (stock data)
    df_ozet = scrape_stock_data()

    if df_ozet is None:
        print("Failed to create df_ozet. Exiting.")
        return

    # Scrape the second dataset (circulation data)
    df_dolasim = scrape_circulation_data()

    # Merge the dataframes
    df_ozet = merge_dataframes(df_ozet, df_dolasim)

    # Add date column to df_ozet
    df_ozet = add_date_column(df_ozet)

    if df_ozet is not None:
        print("\n" + "=" * 50)
        print("FIRST 5 ROWS OF EXPANDED df_ozet:")
        print("=" * 50)
        print(df_ozet.head())

        print(f"\nFinal DataFrame Info:")
        print(f"Shape: {df_ozet.shape}")
        print(f"Columns: {list(df_ozet.columns)}")

        # Export to Excel with date in filename
        export_success = export_to_excel(df_ozet)

        if export_success:
            workday = get_last_workday()
            date_str = workday.strftime('%Y%m%d')
            print(f"\n✓ Data successfully exported to df_ozet_{date_str}.xlsx")
        else:
            print("\n✗ Failed to export data to Excel")

    else:
        print("Failed to create final DataFrame.")


if __name__ == "__main__":
    main()