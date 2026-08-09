import pandas as pd
from config import DATA_DIR
# 1. Define your vertical stock parameters
TICKERS = ['amat', 'avgo','amd','intc','mrvl','mu','tsm', 'nvda']
BAR_SIZE = '4hours'
FILE_SUFFIX = 'extended_feng'

# 2. Define your horizontal context file list
CONTEXT_TICKERS = ['spy', 'smh']

# --- Dynamically generate paths ---

# Vertical input files list
INPUT_FILES = []
for ticker in TICKERS:
    filename = f"{DATA_DIR}/semiconductor/{ticker}_{BAR_SIZE}_{FILE_SUFFIX}.csv"
    INPUT_FILES.append({
        "ticker": ticker,
        "path": filename,
        "enabled": True
    })

# Context input files list
CONTEXT_FILES = []
for ctx_ticker in CONTEXT_TICKERS:
    filename = f"{DATA_DIR}/context/{ctx_ticker}_{BAR_SIZE}_{FILE_SUFFIX}.csv"
    CONTEXT_FILES.append({
        "context_ticker": ctx_ticker,
        "path": filename,
        "enabled": True
    })

def validate_unique_datetime(df, identifier):
    """
    Checks if the 'datetime' column has any duplicate entries for a given dataset.
    Raises a ValueError if duplicates are found.
    """
    if 'datetime' not in df.columns:
        raise KeyError(f"Dataset '{identifier}' is missing the required 'datetime' column.")
        
    # Check for duplicate timestamps
    if df['datetime'].duplicated().any():
        duplicate_count = df['datetime'].duplicated().sum()
        raise ValueError(
            f"ERROR: Duplicate datetime values found in '{identifier}'! "
            f"Found {duplicate_count} duplicate observation(s). "
            f"Each datetime must be completely unique per file."
        )

def load_and_merge_stocks(file_list, context_list, output_path):
    """
    1. Validates and vertically merges primary stock files.
    2. Validates and iterates through multiple context files, prefixing columns 
       and merging them horizontally on datetime.
    3. Encodes ticker IDs and sorts by datetime, ticker, and group.
    """
    df_list = []
    
    # --- STEP A: Vertical Merge & Validation ---
    active_files = [item for item in file_list if item.get("enabled", True)]
    if not active_files:
        raise ValueError("No active vertical CSV files selected for merging.")
        
    for item in active_files:
        file_path = item["path"]
        print(f"Loading & validating vertical file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Standardize datetime format before validation
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
        # Validate uniqueness for this specific vertical ticker file
        ticker_name = item.get("ticker", file_path)
        validate_unique_datetime(df, f"Vertical Ticker: {ticker_name}")
        
        df_list.append(df)
        
    combined_df = pd.concat(df_list, ignore_index=True)
        
    # --- STEP B: Horizontal Context Merges & Validation ---
    active_contexts = [item for item in context_list if item.get("enabled", True)]
    
    for ctx_item in active_contexts:
        ctx_path = ctx_item["path"]
        ctx_prefix = ctx_item["context_ticker"]
        
        print(f"Loading & validating horizontal context file: {ctx_path}")
        context_df = pd.read_csv(ctx_path)
        
        # Standardize datetime format before validation
        if 'datetime' in context_df.columns:
            context_df['datetime'] = pd.to_datetime(context_df['datetime'])
            
        # Validate uniqueness for this specific context file
        validate_unique_datetime(context_df, f"Context Ticker: {ctx_prefix}")
            
        # Drop 'ticker' and 'group' columns from context
        cols_to_drop = [col for col in ['ticker', 'group'] if col in context_df.columns]
        context_df = context_df.drop(columns=cols_to_drop)
        
        # Rename all columns (except datetime) to include the specific context prefix
        rename_mapping = {
            col: f"{ctx_prefix}_{col}" 
            for col in context_df.columns if col != 'datetime'
        }
        context_df = context_df.rename(columns=rename_mapping)
        
        # Merge horizontally using a left join on 'datetime'
        combined_df = pd.merge(combined_df, context_df, on='datetime', how='left')
        
    # --- STEP C: Encoding & Sorting ---
    # Create an encoded 'ticker_id' based on the 'ticker' column
    combined_df['ticker_id'] = combined_df['ticker'].astype('category').cat.codes
    
    # Sort by datetime, ticker, and group
    sort_columns = ['datetime', 'ticker', 'group']
    existing_sort_cols = [col for col in sort_columns if col in combined_df.columns]
    
    if existing_sort_cols:
        combined_df = combined_df.sort_values(by=existing_sort_cols).reset_index(drop=True)
        
    # --- STEP D: Save Output ---
    combined_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully validated, merged, and saved to {output_path}")
    
    # Display the ticker encoding mapping
    print("\nTicker ID Mapping:")
    mapping = combined_df[['ticker', 'ticker_id']].drop_duplicates().reset_index(drop=True)
    print(mapping)
    first_cols = ['datetime', 'ticker', 'ticker_id','group','smh_ReturnZ','spy_ReturnZ']
    other_cols = [c for c in combined_df.columns if c not in first_cols]
    combined_df = combined_df[first_cols + other_cols]
    return combined_df

# Run the merge process
merge_df = load_and_merge_stocks(
    file_list=INPUT_FILES, 
    context_list=CONTEXT_FILES, 
    output_path=f"{DATA_DIR}/combined_semiconductor_{BAR_SIZE}_{FILE_SUFFIX}.csv"
)