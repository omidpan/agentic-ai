import os
import argparse
import pandas as pd
from config import DATA_DIR

# 1. Initialize the parser
parser = argparse.ArgumentParser(description="Process a stock symbol.")
parser.add_argument("-s" ,"--symbol",
                    type=str,
                    required=False,
                    help="The stock symbol to process. default is NVDA.", 
                    nargs='?', default="NVDA")
parser.add_argument("-bs", "--bar_size",
                    type=str,
                    required=False,
                    help="candle size of historical data. default is 1 hour.", 
                    nargs='?', default="1 hour")
args = parser.parse_args()
bar_size = args.bar_size
symbol = args.symbol.lower()

def merge_daily_and_overnight(symbol, bar_size):
    clean_bar_name = str(bar_size).replace(" ", "").lower()
    symbol_lower = str(symbol).lower()
    
    daily_file = f"{DATA_DIR}/{symbol_lower}_{clean_bar_name}_init.csv"
    overnight_file = f"{DATA_DIR}/{symbol_lower}_{clean_bar_name}_overnight.csv" 
    output_file = f"{DATA_DIR}/{symbol_lower}_{clean_bar_name}.csv"
    
    if not os.path.exists(daily_file):
        print(f"Error: Daily data file not found at {daily_file}")
        return
    if not os.path.exists(overnight_file):
        print(f"Error: Overnight data file not found at {overnight_file}")
        return
        
    print(f"Loading data for {symbol.upper()}...")
    df_daily = pd.read_csv(daily_file)
    df_overnight = pd.read_csv(overnight_file)
    
    df_daily["session"] = "daily"
    df_overnight["session"] = "overnight"
    
    # 2. Convert to datetime objects
    df_daily["datetime"] = pd.to_datetime(df_daily["datetime"])
    df_overnight["datetime"] = pd.to_datetime(df_overnight["datetime"])
    
    # 3. Filter overnight rows based on the calendar date (in Eastern Time)
    valid_daily_dates = set(df_daily["datetime"].dt.date.unique())
    df_overnight_filtered = df_overnight[df_overnight["datetime"].dt.date.isin(valid_daily_dates)].copy()
    
    # 4. Concatenate dataframes
    print("Merging dataframes...")
    combined_df = pd.concat([df_daily, df_overnight_filtered], ignore_index=True)
    
    # 5. Sort chronologically
    combined_df = combined_df.sort_values(by=["datetime", "session"]).reset_index(drop=True)
    
    # 6. TIME ZONE FIX: Explicitly localize to New York time (EST/EDT)
    print("Localizing timestamps to America/New_York...")
    combined_df["datetime"] = (
        combined_df["datetime"]
        .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    )
    
    combined_df = combined_df.dropna(subset=["datetime"]).reset_index(drop=True)
    
    # 7. Convert Datetime to Epoch Timestamp (Seconds)
    combined_df["datetime"] = (combined_df["datetime"].astype("int64") // 10**9)
    
    # 8. DROP DUPLICATE ROWS (Fixes the duplicate overnight rows bug)
    # This checks both 'datetime' and 'session' to ensure we only remove true duplicates
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["datetime", "session"], keep="first").reset_index(drop=True)
    removed_duplicates = initial_len - len(combined_df)
    print(f"Removed {removed_duplicates} identical duplicate session rows.")
    
    # 9. Save the master file
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nMerge Complete for {symbol.upper()}:")
    print(f"- Daily bars loaded: {len(df_daily)}")
    print(f"- Overnight bars kept: {len(df_overnight_filtered)}")
    print(f"- Total combined bars preserved: {len(combined_df)}")
    print(f"Saved master file to: {output_file}")
    
    return combined_df

if __name__ == "__main__":
    merge_daily_and_overnight(symbol, bar_size)


# import os
# import argparse
# import pandas as pd
# from config import DATA_DIR

# # 1. Initialize the parser
# parser = argparse.ArgumentParser(description="Process a stock symbol.")
# parser.add_argument("-s" ,"--symbol",
#                     type=str,
#                     required=False,
#                     help="The stock symbol to process. default is NVDA.", 
#                     nargs='?', default="NVDA")
# parser.add_argument("-bs", "--bar_size",
#                     type=str,
#                     required=False,
#                     help="candle size of historical data. default is 1 hour.", 
#                     nargs='?', default="1 hour")
# args = parser.parse_args()
# bar_size = args.bar_size
# symbol = args.symbol.lower()

# def merge_daily_and_overnight(symbol, bar_size):
#     clean_bar_name = str(bar_size).replace(" ", "").lower()
#     symbol_lower = str(symbol).lower()
    
#     daily_file = f"{DATA_DIR}/{symbol_lower}_{clean_bar_name}_init.csv"
#     overnight_file = f"{DATA_DIR}/{symbol_lower}_{clean_bar_name}_overnight.csv" 
#     output_file = f"{DATA_DIR}/{symbol_lower}_{clean_bar_name}.csv"
    
#     if not os.path.exists(daily_file):
#         print(f"Error: Daily data file not found at {daily_file}")
#         return
#     if not os.path.exists(overnight_file):
#         print(f"Error: Overnight data file not found at {overnight_file}")
#         return
        
#     print(f"Loading data for {symbol.upper()}...")
#     df_daily = pd.read_csv(daily_file)
#     df_overnight = pd.read_csv(overnight_file)
    
#     df_daily["session"] = "daily"
#     df_overnight["session"] = "overnight"
    
#     # 2. Convert to datetime objects
#     df_daily["datetime"] = pd.to_datetime(df_daily["datetime"])
#     df_overnight["datetime"] = pd.to_datetime(df_overnight["datetime"])
    
#     # 3. Filter overnight rows based on the calendar date (in Eastern Time)
#     valid_daily_dates = set(df_daily["datetime"].dt.date.unique())
#     df_overnight_filtered = df_overnight[df_overnight["datetime"].dt.date.isin(valid_daily_dates)].copy()
    
#     # 4. Concatenate dataframes
#     print("Merging dataframes...")
#     combined_df = pd.concat([df_daily, df_overnight_filtered], ignore_index=True)
    
#     # 5. Sort chronologically
#     combined_df = combined_df.sort_values(by=["datetime", "session"]).reset_index(drop=True)
    
#     # 6. TIME ZONE FIX: Explicitly localize to New York time (EST/EDT)
#     # dt.tz_localize anchors the naive string to NY time. 
#     # dt.tz_convert(None) or casting to int64 then converts it cleanly to standard UTC Unix Epoch.
#     print("Localizing timestamps to America/New_York for accurate Epoch calculation...")
#     combined_df["datetime"] = (
#         combined_df["datetime"]
#         .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
#     )
    
#     # Drop rows that failed time zone localization (e.g. spring forward gaps)
#     combined_df = combined_df.dropna(subset=["datetime"]).reset_index(drop=True)
    
#     # 7. Convert Datetime to Epoch Timestamp (Seconds)
#     combined_df["datetime"] = (combined_df["datetime"].astype("int64") // 10**9)
    
#     # 8. Save the master file
#     combined_df.to_csv(output_file, index=False)
    
#     print(f"\nMerge Complete for {symbol.upper()}:")
#     print(f"- Daily bars loaded: {len(df_daily)}")
#     print(f"- Overnight bars kept: {len(df_overnight_filtered)}")
#     print(f"- Total combined bars preserved: {len(combined_df)}")
#     print(f"- Verification - Epoch for '2026-07-30 03:00:00' should be 1785481200: {combined_df['datetime'].iloc[-1]}")
#     print(f"Saved master file to: {output_file}")
    
#     return combined_df

# if __name__ == "__main__":
#     merge_daily_and_overnight(symbol, bar_size)

