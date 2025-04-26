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

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Stock Correlation Matrix")

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
        raw_data = yf.download(tickers, start=fetch_start_date, end=fetch_end_date, interval='1d', progress=False, ignore_tz=True)
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
    if not batch_success: # Individual Download Fallback
        log_messages.append("INFO: Attempting individual downloads...")
        all_data_individual = {}; valid_tickers_individual = []; failed_tickers_individual = []; warnings_individual = {}
        for ticker in tickers:
             processed = False
             try:
                 time.sleep(0.02); ticker_data = yf.download(ticker, start=fetch_start_date, end=fetch_end_date, interval='1d', progress=False, ignore_tz=True)
                 if ticker_data.empty: failed_tickers_individual.append(ticker); continue
                 if 'Adj Close' in ticker_data.columns:
                     adj_close_col = ticker_data['Adj Close'];
                     if isinstance(adj_close_col, pd.Series) and not adj_close_col.isnull().all(): all_data_individual[ticker] = adj_close_col; valid_tickers_individual.append(ticker); processed = True
                 if not processed and 'Close' in ticker_data.columns:
                     close_col = ticker_data['Close'];
                     if isinstance(close_col, pd.Series) and not close_col.isnull().all(): warnings_individual[ticker] = "Using 'Close' price."; all_data_individual[ticker] = close_col; valid_tickers_individual.append(ticker); processed = True
                 if not processed: failed_tickers_individual.append(ticker)
             except Exception as e: warnings_individual[ticker] = f"Error fetching/processing: {str(e)}"; failed_tickers_individual.append(ticker)
        if not valid_tickers_individual: log_messages.append("ERROR: Individual downloads failed for all tickers."); return None, [], log_messages
        log_messages.append(f"SUCCESS: Processed individual data for: {', '.join(sorted(valid_tickers_individual))}")
        for ticker, msg in warnings_individual.items(): log_messages.append(f"WARNING: {ticker}: {msg}")
        if failed_tickers_individual: log_messages.append(f"WARNING: Failed individual download/processing for: {', '.join(sorted(list(set(failed_tickers_individual))))}")
        try: processed_data = pd.DataFrame(all_data_individual);
        except Exception as e: log_messages.append(f"ERROR: Error combining individual data into DataFrame: {str(e)}"); return None, [], log_messages
        if processed_data.empty: log_messages.append("ERROR: Failed to combine individual data (DataFrame empty)."); return None, [], log_messages
    if processed_data is None or processed_data.empty: log_messages.append("ERROR: No usable stock data could be retrieved after all attempts."); return None, [], log_messages
    try: # Final Post-Processing
        log_messages.append("INFO: Post-processing final data...")
        if not isinstance(processed_data.index, pd.DatetimeIndex): processed_data.index = pd.to_datetime(processed_data.index)
        processed_data = processed_data.sort_index(); processed_data = processed_data.ffill().bfill()
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
    except Exception as e: log_messages.append(f"ERROR: Error during final data processing: {str(e)}"); log_messages.append(f"Traceback: {traceback.format_exc()}"); return None, [], log_messages

# --- Initialize Session State ---
if 'run_analysis_clicked' not in st.session_state: st.session_state.run_analysis_clicked = False
if 'tickers_for_analysis' not in st.session_state: st.session_state.tickers_for_analysis = []
if 'fetch_log_messages' not in st.session_state: st.session_state.fetch_log_messages = []
# --- !!! Set cluster_map default to True !!! ---
if 'cluster_map' not in st.session_state: st.session_state.cluster_map = True

end_date_default = MAX_FETCH_END_DATE
start_date_default = max(MAX_FETCH_START_DATE, end_date_default - timedelta(days=365))
def initialize_or_validate_date(state_key, default_value, min_val, max_val):
    if state_key not in st.session_state: st.session_state[state_key] = default_value
    else:
        try: current_val = pd.to_datetime(st.session_state[state_key]).date(); st.session_state[state_key] = max(min_val, min(max_val, current_val))
        except Exception: st.session_state[state_key] = default_value
initialize_or_validate_date('start_date', start_date_default, MAX_FETCH_START_DATE, MAX_FETCH_END_DATE)
initialize_or_validate_date('end_date', end_date_default, MAX_FETCH_START_DATE, MAX_FETCH_END_DATE)
if st.session_state.start_date >= st.session_state.end_date: st.session_state.start_date = max(MAX_FETCH_START_DATE, st.session_state.end_date - timedelta(days=1))

# --- Callback Functions for Preset Buttons ---
def set_date_range(months=None, years=None, max_range=False):
    end = MAX_FETCH_END_DATE; start = MAX_FETCH_START_DATE
    if max_range: start = MAX_FETCH_START_DATE
    elif years:
        try: start = max(MAX_FETCH_START_DATE, end.replace(year=end.year - years))
        except ValueError: start = max(MAX_FETCH_START_DATE, end - timedelta(days=365 * years))
    elif months: start = max(MAX_FETCH_START_DATE, end - timedelta(days=int(30.5 * months)))
    if start > end: start = end
    start = max(MAX_FETCH_START_DATE, start)
    st.session_state.start_date = start; st.session_state.end_date = end
    st.session_state.run_analysis_clicked = False; st.session_state.fetch_log_messages = []

# --- Function to convert df to csv for download ---
@st.cache_data
def convert_df_to_csv(df):
  try: return df.to_csv().encode('utf-8')
  except Exception as e: st.error(f"Error converting data to CSV: {e}"); return None

# --- Streamlit App UI ---
st.title("Stock Correlation Matrix Analyzer")
st.caption("Visualize how closely the daily returns of different stocks moved together over a selected time period.")
st.markdown("---")

# --- Inputs ---
col1, col2 = st.columns([1, 2]) # Input column (33%), Output column (67%)

with col1:
    st.subheader("Inputs")
    # --- !!! UPDATED DEFAULT TICKERS !!! ---
    default_tickers_str = "LMND, HIMS, SOFI, UPST, NVDA, PLTR, TSLA, HOOD, COIN, GOOG, AMZN, NU"
    # --- !!! END UPDATED TICKERS !!! ---
    ticker_input_str = st.text_area(
        "1. Enter Stock Tickers:", value=default_tickers_str, height=150, key="ticker_input",
        help="Separate tickers with spaces, commas, semicolons, or newlines."
    )
    st.markdown("2. Select Date Range:")
    preset_cols = st.columns(5)
    with preset_cols[0]: st.button("3M", on_click=set_date_range, kwargs={'months': 3}, use_container_width=True, key="preset_3m")
    with preset_cols[1]: st.button("6M", on_click=set_date_range, kwargs={'months': 6}, use_container_width=True, key="preset_6m")
    with preset_cols[2]: st.button("1Y", on_click=set_date_range, kwargs={'years': 1}, use_container_width=True, key="preset_1y")
    with preset_cols[3]: st.button("2Y", on_click=set_date_range, kwargs={'years': 2}, use_container_width=True, key="preset_2y")
    with preset_cols[4]: st.button("Max", on_click=set_date_range, kwargs={'max_range': True}, use_container_width=True, key="preset_max")
    st.markdown("Or Select Custom Range:")
    date_col1, date_col2 = st.columns(2)
    with date_col1: st.date_input("Start Date", min_value=MAX_FETCH_START_DATE, max_value=MAX_FETCH_END_DATE, key="start_date")
    with date_col2:
        min_end_allowed_precise = st.session_state.start_date + timedelta(days=1)
        st.date_input("End Date", min_value=min_end_allowed_precise, max_value=MAX_FETCH_END_DATE, key="end_date")

    # --- Display Options ---
    st.markdown("3. Display Options:")
    # --- !!! Set default value for checkbox to True !!! ---
    st.checkbox("Cluster Heatmap (Group Similar Stocks)", key='cluster_map', value=st.session_state.cluster_map)
    # --- !!! END CHECKBOX DEFAULT ---

    st.markdown("---")
    run_button_clicked = st.button("Run Analysis", type="primary", key="run_button")

    # --- Log Display Area ---
    st.markdown("---")
    st.subheader("Data Fetch Log")
    log_placeholder = st.empty()
    if st.session_state.fetch_log_messages:
        log_html = "<br>".join(st.session_state.fetch_log_messages)
        log_placeholder.markdown(f"<div style='font-size: 0.9em; max-height: 150px; overflow-y: auto; border: 1px solid #eee; padding: 5px;'>{log_html}</div>", unsafe_allow_html=True)
    else:
        log_placeholder.info("Logs will appear here after running analysis.")
    # --- END LOG DISPLAY ---

    if run_button_clicked:
        parsed_tickers = parse_tickers(ticker_input_str)
        if not parsed_tickers: st.warning("Please enter at least one valid ticker symbol."); st.session_state.run_analysis_clicked = False; st.session_state.fetch_log_messages = []
        elif st.session_state.start_date >= st.session_state.end_date: st.warning("Error: Start Date must be before End Date."); st.session_state.run_analysis_clicked = False; st.session_state.fetch_log_messages = []
        else: st.session_state.run_analysis_clicked = True; st.session_state.tickers_for_analysis = parsed_tickers; st.session_state.fetch_log_messages = []

# --- Analysis and Plotting ---
with col2:
    st.subheader("Correlation Matrix")
    analysis_placeholder = st.empty()

    if 'correlation_matrix_result' not in st.session_state:
        st.session_state.correlation_matrix_result = None

    if st.session_state.run_analysis_clicked and st.session_state.tickers_for_analysis:
        tickers = st.session_state.tickers_for_analysis
        start_date_selected = st.session_state.start_date
        end_date_selected = st.session_state.end_date
        corr_method = 'pearson' # Hardcoded
        cluster_map = st.session_state.cluster_map # Get cluster option from state

        with analysis_placeholder.container():
            with st.spinner(f"Running analysis for {len(tickers)} tickers..."):
                full_price_df, fetched_tickers, log_msgs = fetch_max_stock_data(tuple(tickers))
                st.session_state.fetch_log_messages = log_msgs
                log_html = "<br>".join(st.session_state.fetch_log_messages)
                log_placeholder.markdown(f"<div style='font-size: 0.9em; max-height: 150px; overflow-y: auto; border: 1px solid #eee; padding: 5px;'>{log_html}</div>", unsafe_allow_html=True)

                if not isinstance(fetched_tickers, list): fetched_tickers = []
                if full_price_df is None or full_price_df.empty: st.error("Failed to fetch base data. Check logs below inputs for details."); st.stop()

                needed_tickers_set = set(tickers); available_tickers_set = set(fetched_tickers)
                if not needed_tickers_set.issubset(available_tickers_set):
                    missing = sorted(list(needed_tickers_set - available_tickers_set)); st.error(f"Failed to fetch base data for all requested tickers.")
                    if missing: st.info(f"Missing after fetch: {', '.join(missing)}"); st.stop()

                try: # Filter Data
                    if not isinstance(full_price_df.index, pd.DatetimeIndex): full_price_df.index = pd.to_datetime(full_price_df.index)
                    mask = (full_price_df.index.date >= start_date_selected) & (full_price_df.index.date <= end_date_selected)
                    columns_to_select = [t for t in tickers if t in full_price_df.columns]; price_df_filtered = full_price_df.loc[mask, columns_to_select]
                except Exception as filter_err: st.error(f"Error filtering data: {str(filter_err)}"); st.error(f"Traceback: {traceback.format_exc()}"); st.stop()

                if price_df_filtered.empty: st.warning(f"No data available for the selected range ({start_date_selected} to {end_date_selected})."); st.stop()

                returns_df = price_df_filtered.pct_change().iloc[1:]
                if returns_df.empty: st.warning("Could not calculate returns."); st.stop()

                returns_df_cleaned_any = returns_df.dropna(how='any'); num_days = len(returns_df_cleaned_any)

                if num_days >= MIN_DATA_POINTS_FOR_CORR:
                    std_dev = returns_df_cleaned_any.std()
                    if (std_dev < 1e-10).any():
                        zero_var_tickers = std_dev[(std_dev < 1e-10)].index.tolist(); st.warning(f"Cannot calc correlation: Ticker(s) {', '.join(zero_var_tickers)} had zero variance."); st.stop()
                    else:
                        corr_matrix = returns_df_cleaned_any.corr(method=corr_method)
                        st.session_state.correlation_matrix_result = corr_matrix

                        # --- Clustering (Optional) ---
                        plot_order = corr_matrix.columns # Default order
                        if cluster_map and len(corr_matrix.columns) > 2:
                            try:
                                dists = 1 - corr_matrix.values # Distance = 1 - correlation
                                linkage = hierarchy.linkage(dists, method='average')
                                dendro = hierarchy.dendrogram(linkage, no_plot=True)
                                plot_order = [corr_matrix.columns[i] for i in dendro['leaves']]
                            except Exception as cluster_err:
                                st.warning(f"Clustering failed: {cluster_err}. Displaying in original order.")
                                plot_order = corr_matrix.columns

                        # Reindex matrix for plotting based on plot_order
                        corr_matrix_display = corr_matrix.loc[plot_order, plot_order]
                        # --- END CLUSTERING ---

                        corr_matrix_plot = corr_matrix_display.copy(); np.fill_diagonal(corr_matrix_plot.values, np.nan)

                        custom_colorscale_refined = [
                            [0.0,   'rgb(178,24,43)'],    # Dark Red (-1.0)
                            [0.6,   'rgb(253,174,97)'],   # Orange-ish (at 0.2)
                            [0.7,   'rgb(248, 248, 248)'],# Off-white (at 0.4 - neutral center)
                            [0.9,   'rgb(116,173,209)'],  # Medium Blue (at 0.8)
                            [1.0,   'rgb(5,48,97)']       # Dark Blue (1.0)
                        ]

                        try: # Plotting Block
                            fig_corr = px.imshow(
                                corr_matrix_plot, text_auto=".2f", aspect="equal",
                                color_continuous_scale=custom_colorscale_refined,
                                zmin=-1, zmax=1,
                                title=f"Daily Return Correlation ({start_date_selected} to {end_date_selected})"
                            )
                            fig_corr.update_layout(height=min(700, 45 * len(corr_matrix.columns) + 150), coloraxis_colorbar=dict(title="Corr"), xaxis_title=None, yaxis_title=None, xaxis={'side': 'bottom'})
                            fig_corr.update_xaxes(tickangle=45); fig_corr.update_yaxes(ticklabelposition="outside"); fig_corr.update_traces(xgap=1, ygap=1)

                            st.plotly_chart(fig_corr, use_container_width=True)
                            st.caption(f"Correlation matrix based on **{num_days}** trading days where all stocks had valid returns.")

                            # --- ADDED LEGEND/EXPLANATION ---
                            st.markdown("""
                            <style>
                            .legend-item { margin-bottom: 5px; display: flex; align-items: center; font-size: 0.9em; }
                            .legend-color-box { width: 15px; height: 15px; margin-right: 8px; border: 1px solid #ccc; display: inline-block; flex-shrink: 0; border-radius: 3px; }
                            </style>
                            <div style="border: 1px solid #eee; padding: 10px; border-radius: 5px; margin-top: 15px;">
                                <strong style="display: block; margin-bottom: 8px;">Interpreting the Colors:</strong>
                                <div class="legend-item">
                                    <span class="legend-color-box" style="background-color:rgb(5,48,97);"></span> <strong>High Positive (e.g., > 0.8):</strong> Returns strongly move together.
                                </div>
                                <div class="legend-item">
                                    <span class="legend-color-box" style="background-color:rgb(116,173,209);"></span> <strong>Moderate Positive (e.g., 0.4 to 0.8):</strong> Returns tend to move together.
                                </div>
                                <div class="legend-item">
                                    <span class="legend-color-box" style="background-color:rgb(248, 248, 248);"></span> <strong>Low / Neutral (e.g., 0.2 to 0.4):</strong> Weak linear relationship.
                                </div>
                                <div class="legend-item">
                                    <span class="legend-color-box" style="background-color:rgb(253,174,97);"></span>
                                    <span class="legend-color-box" style="background-color:rgb(178,24,43); margin-left: -10px;"></span> <strong>Low Positive / Negative (&lt; 0.2):</strong> Returns mostly independent or move oppositely.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("---") # Separator before download button
                            # --- END LEGEND ---

                            # --- Download Button ---
                            csv_data = convert_df_to_csv(corr_matrix) # Download original (unclustered) matrix
                            if csv_data:
                                st.download_button(
                                    label="Download Correlation Matrix (CSV)",
                                    data=csv_data,
                                    file_name=f"correlation_matrix_{start_date_selected}_to_{end_date_selected}.csv",
                                    mime='text/csv',
                                    key="download_csv_button"
                                )

                        except Exception as plot_err: st.error(f"An error occurred during plotting: {str(plot_err)}"); st.error(f"Traceback: {traceback.format_exc()}")
                else: # Insufficient data
                    st.warning(f"Insufficient overlapping data ({num_days} days) within the selected date range ({start_date_selected} to {end_date_selected}). Need at least {MIN_DATA_POINTS_FOR_CORR} days.")
                    if not returns_df_cleaned_any.empty:
                        with st.expander(f"Show {num_days} Day(s) of Overlapping Returns"): st.dataframe(returns_df_cleaned_any.head())
                    elif not price_df_filtered.empty:
                         with st.expander("Show Filtered Price Data (Head)"): st.dataframe(price_df_filtered.head())
                    st.session_state.correlation_matrix_result = None
    else:
        analysis_placeholder.info("Enter tickers, select a date range, and click 'Run Analysis'.")
        st.session_state.correlation_matrix_result = None

# --- Explainer Section ---
st.markdown("---")
with st.expander("How is the Correlation Calculated?"):
    st.markdown(
        """
        The correlation matrix shows how closely the daily price movements of different stocks align with each other over your selected time period. Here's a breakdown of the steps:

        1.  **Data Fetching:** The app retrieves historical daily price data (preferring Adjusted Close prices, which account for dividends and stock splits, but falling back to Close prices if necessary) for your selected tickers from Yahoo Finance. It fetches up to 5 years of data initially.
        2.  **Date Filtering:** Only the data within the specific Start and End Dates you selected (using presets or date inputs) is used for the calculation.
        3.  **Daily Return Calculation:** For each stock, the app calculates the daily percentage return: `(Today's Price - Yesterday's Price) / Yesterday's Price`.
        4.  **Data Alignment:** To compare stocks fairly, only the trading days where *all* selected stocks have a valid return value are kept. If any stock has missing data on a particular day, that entire day is excluded from the calculation for *all* pairs.
        5.  **Pearson Correlation Calculation:** Using the aligned set of daily returns, the app calculates the **Pearson correlation coefficient** between every possible pair of stocks. This value ranges from **-1.0 (perfectly opposite movement) to +1.0 (perfectly synchronized movement)**, with 0.0 indicating no linear relationship.
        6.  **Visualization:** The heatmap displays these coefficients. The diagonal is ignored (NaN) as a stock's correlation with itself is always +1.0. The matrix can optionally be clustered to group similar stocks.

        *Note: Correlation is specific to the chosen date range and can change significantly over different periods.*
        """
    )

# --- Legal Disclaimer Section ---
st.markdown("---")
with st.expander("Legal Disclaimer"):
    st.markdown(
         """
         This site does not provide registered investment advice and is not a broker/dealer. The materials and information accessible here should be used for informational purposes only. No information constitutes a recommendation that any particular investment, security, portfolio of securities, transaction or investment strategy is suitable for any specific person.

         You should not rely solely upon the information provided by this tool to make an investment decision. You should always conduct your own research and due diligence and obtain professional advice before making any investment decision. You hereby acknowledge and agree that this site is not operated as an offer to, or solicitation of, any potential clients or investors for the provision of investment management, advisory or any other service. You agree not to construe any content or materials listed on this site as tax, legal, insurance or investment advice or as an offer to sell, or as a solicitation of an offer to buy, any security or other financial instrument.

         This site and its creators will not be liable for any loss or damage caused by reliance on any information obtained through or from this tool. No guarantee can be made that you will profit off the information provided. While best efforts are made to ensure data accuracy using reliable sources (like Yahoo Finance), we cannot guarantee that all the data presented is accurate. Please understand this, and it’s always a good practice to double-check numbers against the companies' own financial documents or other primary sources.
         """
    )

# --- Footer ---
st.markdown("---")
st.caption(f"Data sourced from Yahoo Finance via yfinance. Fetched/Analyzed on: {TODAY.strftime('%Y-%m-%d')}")
