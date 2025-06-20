import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta, date
import re
import time
import traceback
from scipy.cluster import hierarchy # For clustering
import random # <--- ADDED: For randomized delays
import requests_cache # <--- ADDED: For robust HTTP caching

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Stock Correlation Matrix")

# --- Robustness Setup ---
# <--- ADDED: Setup a cache for web requests. Data will be stored in 'yfinance.cache'
# and will expire after 1 hour. This prevents re-hitting the API for the same
# data within a short period, even across app restarts.
session = requests_cache.CachedSession('yfinance.cache', expire_after=timedelta(hours=1))
session.headers['User-agent'] = 'my-stock-app/1.0' # <--- ADDED: Identify your app

# --- Constants ---
MIN_DATA_POINTS_FOR_CORR = 30
MAX_HISTORY_YEARS = 5
TODAY = datetime.now().date()
MARKET_CLOSE_LAG_DAYS = 1
potential_end_date = TODAY - timedelta(days=MARKET_CLOSE_LAG_DAYS)
MAX_FETCH_END_DATE = min(TODAY, potential_end_date)
MAX_FETCH_START_DATE = MAX_FETCH_END_DATE - timedelta(days=365 * MAX_HISTORY_YEARS)

# --- Helper Function for Ticker Parsing ---
def parse_tickers(ticker_string):
    """Parses a string of tickers into a list, handling various delimiters."""
    if not ticker_string: return []
    tickers = re.split(r'[,\s\n;]+', ticker_string)
    cleaned_tickers = [t.strip().upper() for t in tickers if t.strip()]
    return sorted(list(set(cleaned_tickers)))

# --- Caching Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_max_stock_data(tickers_tuple):
    """
    Fetches up to MAX_HISTORY_YEARS of data for tickers.
    Returns: DataFrame, list of fetched tickers, list of log messages.
    """
    tickers = list(tickers_tuple); log_messages = []
    if not tickers: return None, [], ["Error: No tickers provided."]
    log_messages.append(f"INFO: Fetching max history ({MAX_HISTORY_YEARS} years) for {len(tickers)} tickers (if not cached)...")
    fetch_start_date = MAX_FETCH_START_DATE; fetch_end_date = MAX_FETCH_END_DATE
    processed_data = None; batch_success = False; price_type_found = None
    failed_tickers_batch = []
    try: # Batch Download Attempt
        # <--- CHANGED: Pass the cached session to yfinance
        raw_data = yf.download(
            tickers,
            start=fetch_start_date,
            end=fetch_end_date,
            interval='1d',
            progress=False,
            ignore_tz=True,
            session=session # <--- ADDED
        )
        if not raw_data.empty:
            temp_data = None
            if isinstance(raw_data.columns, pd.MultiIndex):
                levels = raw_data.columns.levels[0];
                if 'Adj Close' in levels: price_type_found = 'Adj Close'
                elif 'Close' in levels: price_type_found = 'Close'; log_messages.append("WARNING: Using 'Close' prices (batch).")
                if price_type_found: temp_data = raw_data[price_type_found]
            elif isinstance(raw_data.columns, pd.Index):
                    if 'Adj Close' in raw_data.columns: price_type_found = 'Adj Close'; temp_data = raw_data[['Adj Close']]
                    elif 'Close' in raw_data.columns: price_type_found = 'Close'; temp_data = raw_data[['Close']]; log_messages.append("WARNING: Using 'Close' prices (batch).")
                    if len(tickers) == 1 and price_type_found and temp_data is not None and len(temp_data.columns) == 1: temp_data = temp_data.rename(columns={price_type_found: tickers[0]})
            if temp_data is not None and not temp_data.empty:
                    processed_data = temp_data.dropna(axis=1, how='all')
                    if not processed_data.empty: log_messages.append(f"SUCCESS: Batch download successful using '{price_type_found}'."); batch_success = True
                    else: log_messages.append("WARNING: Batch download successful, but all columns were NaN."); processed_data = None
            else: log_messages.append("WARNING: Batch download ran, but couldn't extract 'Adj Close' or 'Close'.")
            if processed_data is not None:
                failed_tickers_batch = sorted(list(set(tickers) - set(processed_data.columns)))
                if failed_tickers_batch: log_messages.append(f"WARNING: Batch download might be missing data for: {', '.join(failed_tickers_batch)}")
        else: log_messages.append("WARNING: Batch download returned an empty DataFrame.")
    except Exception as e: log_messages.append(f"WARNING: Batch download attempt failed: {str(e)}"); processed_data = None; batch_success = False
    
    # <--- CHANGED: Use the 'failed_tickers_batch' list for the fallback if it exists, otherwise use all tickers
    tickers_for_fallback = failed_tickers_batch if (batch_success and failed_tickers_batch) else tickers
    
    if not batch_success or failed_tickers_batch:
        log_messages.append("INFO: Attempting individual downloads for failed/missing tickers...")
        all_data_individual = {}; valid_tickers_individual = []; failed_tickers_individual = []; warnings_individual = {}

        for ticker in tickers_for_fallback:
            processed = False
            try:
                # <--- CHANGED: Increased and randomized the delay to avoid rate limiting.
                time.sleep(random.uniform(0.2, 0.6))
                ticker_data = yf.download(
                    ticker,
                    start=fetch_start_date,
                    end=fetch_end_date,
                    interval='1d',
                    progress=False,
                    ignore_tz=True,
                    session=session # <--- ADDED
                )
                if ticker_data.empty: failed_tickers_individual.append(ticker); continue
                
                if 'Adj Close' in ticker_data.columns:
                    adj_close_col = ticker_data['Adj Close'];
                    if isinstance(adj_close_col, pd.Series) and not adj_close_col.isnull().all():
                        all_data_individual[ticker] = adj_close_col; valid_tickers_individual.append(ticker); processed = True
                
                if not processed and 'Close' in ticker_data.columns:
                    close_col = ticker_data['Close'];
                    if isinstance(close_col, pd.Series) and not close_col.isnull().all():
                        warnings_individual[ticker] = "Using 'Close' price."; all_data_individual[ticker] = close_col; valid_tickers_individual.append(ticker); processed = True
                
                if not processed: failed_tickers_individual.append(ticker)
            except Exception as e:
                warnings_individual[ticker] = f"Error fetching/processing: {str(e)}"; failed_tickers_individual.append(ticker)

        if valid_tickers_individual:
            log_messages.append(f"SUCCESS: Processed individual data for: {', '.join(sorted(valid_tickers_individual))}")
            for ticker, msg in warnings_individual.items(): log_messages.append(f"WARNING: {ticker}: {msg}")
            if failed_tickers_individual: log_messages.append(f"WARNING: Failed individual download/processing for: {', '.join(sorted(list(set(failed_tickers_individual))))}")
            
            try:
                individual_df = pd.DataFrame(all_data_individual)
                # <--- CHANGED: Merge individual data with existing batch data
                if processed_data is not None:
                    processed_data = pd.concat([processed_data, individual_df], axis=1)
                else:
                    processed_data = individual_df
            except Exception as e:
                log_messages.append(f"ERROR: Error combining data into DataFrame: {str(e)}")
                # If there's an error here but we had batch data, we can still return it
                if processed_data is None: return None, [], log_messages
        elif processed_data is None: # Only fail if both batch and individual downloads yield nothing
            log_messages.append("ERROR: Individual downloads failed for all tickers, and no batch data was available."); return None, [], log_messages
    
    if processed_data is None or processed_data.empty:
        log_messages.append("ERROR: No usable stock data could be retrieved after all attempts.")
        return None, [], log_messages

    try: # Final Post-Processing
        log_messages.append("INFO: Post-processing final data...")
        if not isinstance(processed_data.index, pd.DatetimeIndex): processed_data.index = pd.to_datetime(processed_data.index)
        processed_data = processed_data.sort_index();
        
        # Remove duplicate columns that could appear from batch + individual fetches
        processed_data = processed_data.loc[:,~processed_data.columns.duplicated()]
        
        processed_data = processed_data.ffill().bfill()
        processed_data = processed_data.dropna(axis=1, how='all'); final_tickers_list = list(processed_data.columns)
        if not final_tickers_list: log_messages.append("ERROR: Dataframe became empty after final processing."); return None, [], log_messages

        requested_set = set(tickers); available_set = set(final_tickers_list)
        final_tickers_intersection = sorted(list(requested_set.intersection(available_set)))
        if not final_tickers_intersection:
            log_messages.append("ERROR: None of the requested tickers had valid data after processing.")
            dropped_tickers = sorted(list(available_set - requested_set));
            if dropped_tickers: log_messages.append(f"INFO: Tickers dropped during processing: {', '.join(dropped_tickers)}")
            return None, [], log_messages
        
        processed_data = processed_data[final_tickers_intersection]
        log_messages.append(f"INFO: Max history data ready for {len(final_tickers_intersection)} tickers.")
        return processed_data, final_tickers_intersection, log_messages
    except Exception as e:
        log_messages.append(f"ERROR: Error during final data processing: {str(e)}"); log_messages.append(f"Traceback: {traceback.format_exc()}"); return None, [], log_messages

# --- [The rest of your Streamlit UI and logic code remains the same] ---
# ... (all the code from `if 'run_analysis_clicked' not in st.session_state:` onwards)
# You can copy and paste the rest of your original script here.
# The changes are self-contained within the data fetching section.
# I am omitting the rest of the code for brevity as it does not need to be changed.
