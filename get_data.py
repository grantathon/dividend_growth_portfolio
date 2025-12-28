import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yaml
import yfinance as yf


def setup_logger(log_file_path):
    logger = logging.getLogger('update_data')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger

def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    
    inputs = cfg.get("inputs", {})
    end_date_str = inputs.get("end_date")
    stock_screener_file = inputs.get("stock_screener_file")

    if not end_date_str:
        raise ValueError("Missing 'inputs.end_date' in config.yaml")
    
    if not stock_screener_file:
        raise ValueError("Missing 'inputs.stock_screener_file' in config.yaml")
    
    try:
        end_date_target = pd.to_datetime(end_date_str).date()
    except Exception as exc:
        raise ValueError(f"Invalid end_date format in config: {end_date_str}") from exc
    
    # Calculate start_date target (exactly 5 years prior, same day/month)
    # Use replace to subtract 5 years while keeping same day/month
    # Handle leap year edge case (Feb 29)
    try:
        start_date_target = end_date_target.replace(year=end_date_target.year - 5)
    except ValueError:
        # If Feb 29 doesn't exist 5 years prior, use Feb 28
        start_date_target = end_date_target.replace(year=end_date_target.year - 5, day=28)
    
    # Adjust both dates to nearest previous trading days
    print(f"Target dates: {start_date_target} to {end_date_target}")
    print("Adjusting to nearest previous trading days...")
    
    start_date = find_nearest_trading_day(start_date_target)
    end_date = find_nearest_trading_day(end_date_target)
    
    return start_date.isoformat(), end_date.isoformat(), stock_screener_file

def find_nearest_trading_day(target_date, reference_ticker="SPY", lookback_days=10):
    # Download data around the target date to find trading days
    start_buffer = (target_date - timedelta(days=lookback_days)).isoformat()
    end_buffer = (target_date + timedelta(days=1)).isoformat()
    
    try:
        ticker_yf = yf.Ticker(reference_ticker)
        hist = ticker_yf.history(start=start_buffer, end=end_buffer)
        
        if hist.empty:
            # If no data, try using pandas business days as fallback
            # Go back up to 5 business days to find a trading day
            for i in range(5):
                candidate = target_date - timedelta(days=i)
                if candidate.weekday() < 5:  # Monday=0, Friday=4
                    return candidate
            return target_date - timedelta(days=5)
        
        # Get all trading days from the data
        trading_days = hist.index.date
        
        # Find the nearest trading day <= target_date
        valid_days = [d for d in trading_days if d <= target_date]
        
        if valid_days:
            return max(valid_days)
        else:
            # Fallback: go back a few days
            return target_date - timedelta(days=5)
            
    except Exception as exc:
        print(f"  WARNING: Could not determine trading days using {reference_ticker}: {exc}")
        # Fallback: use business day logic
        for i in range(5):
            candidate = target_date - timedelta(days=i)
            if candidate.weekday() < 5:
                return candidate
        return target_date - timedelta(days=5)

def load_tickers(data_path):
    if not Path(data_path).exists():
        raise FileNotFoundError(f"{data_path} not found")
    
    df = pd.read_csv(data_path)

    # If there's a row where the ticker is "Summary", remove it.
    df = df[df['Ticker'].str.upper() != "SUMMARY"]

    tickers = df['Ticker'].values
    
    return tickers

def download_single_ticker(ticker, start_date, end_date):
    try:
        ticker_yf = yf.Ticker(ticker)
        hist = ticker_yf.history(
            period=None,
            start=start_date,
            end=end_date,
            actions=True,
            auto_adjust=False
        )
        
        if hist.empty:
            return None
        
        # Extract Close and Dividends
        result = pd.DataFrame({
            'Close': hist['Close'],
            'Dividends': hist.get('Dividends', pd.Series(0.0, index=hist.index)).fillna(0.0)
        })
        
        return result
    except Exception as exc:
        raise Exception(f"Failed to download {ticker}: {exc}")

def download_ticker_data_bulk(tickers, start_date, end_date, retry_delay=5, max_retries=3, logger=None):
    print(f"Downloading data for {len(tickers)} tickers...")
    if logger:
        logger.info(f"Downloading data for {len(tickers)} tickers...")
    
    ticker_data = {}
    failed_tickers = []
    
    # First attempt: bulk download
    try:
        print("  Attempting bulk download...")
        tickers_obj = yf.Tickers(" ".join(tickers))
        
        hist = tickers_obj.history(
            period=None,
            start=start_date,
            end=end_date,
            actions=True,
            auto_adjust=False,
            group_by='column',
            progress=True
        )
        
        if not hist.empty and isinstance(hist.columns, pd.MultiIndex):
            # Process successfully downloaded tickers
            for ticker in tickers:
                ticker_upper = ticker.upper()
                close_col = ('Close', ticker_upper)
                div_col = ('Dividends', ticker_upper)
                
                if close_col in hist.columns:
                    close_series = hist[close_col]
                    div_series = hist[div_col] if div_col in hist.columns else pd.Series(0.0, index=hist.index)
                    
                    result = pd.DataFrame({
                        'Close': close_series,
                        'Dividends': div_series.fillna(0.0)
                    })
                    
                    if len(result) > 0:
                        ticker_data[ticker_upper] = result
                        if logger:
                            logger.info(f"  ✓ {ticker_upper}: {len(result)} days of data")
                    else:
                        failed_tickers.append(ticker_upper)
                        if logger:
                            logger.warning(f"  ✗ {ticker_upper}: No data available")
                else:
                    failed_tickers.append(ticker_upper)
                    if logger:
                        logger.warning(f"  ✗ {ticker_upper}: Close column not found in bulk download")
        else:
            # If bulk download failed completely, mark all as failed
            failed_tickers = [t.upper() for t in tickers]
            
    except Exception as exc:
        print(f"  WARNING: Bulk download failed: {exc}")
        # Mark all tickers as failed for individual retry
        failed_tickers = [t.upper() for t in tickers]
    
    # Retry failed tickers individually with delays
    if failed_tickers:
        print(f"\n  Retrying {len(failed_tickers)} failed ticker(s)...")
        if logger:
            logger.info(f"  Retrying {len(failed_tickers)} failed ticker(s)...")
        
        for attempt in range(1, max_retries + 1):
            if not failed_tickers:
                break
                
            print(f"  Retry attempt {attempt}/{max_retries}...")
            if logger:
                logger.info(f"  Retry attempt {attempt}/{max_retries}...")
            still_failed = []
            
            for ticker in failed_tickers:
                try:
                    # Add delay between requests to avoid rate limiting
                    if attempt > 1 or len(failed_tickers) > 1:
                        time.sleep(retry_delay)
                    
                    result = download_single_ticker(ticker, start_date, end_date)
                    
                    if result is not None and len(result) > 0:
                        ticker_data[ticker] = result
                        if logger:
                            logger.info(f"    ✓ {ticker}: {len(result)} days of data")
                    else:
                        still_failed.append(ticker)
                        if logger:
                            logger.warning(f"    ✗ {ticker}: No data available")
                        
                except Exception as exc:
                    still_failed.append(ticker)
                    error_msg = str(exc)
                    if "rate limit" in error_msg.lower() or "429" in error_msg:
                        if logger:
                            logger.warning(f"    ✗ {ticker}: Rate limit hit, will retry")
                    else:
                        if logger:
                            logger.error(f"    ✗ {ticker}: {error_msg}")
            
            failed_tickers = still_failed
            
            if failed_tickers and attempt < max_retries:
                wait_time = retry_delay * 2
                print(f"  Waiting {wait_time} seconds before next retry...")
                if logger:
                    logger.info(f"  Waiting {wait_time} seconds before next retry...")
                time.sleep(wait_time)
    
    # Final summary
    if failed_tickers:
        print(f"\n  WARNING: {len(failed_tickers)} ticker(s) failed after all retries: {', '.join(failed_tickers)}")
        if logger:
            logger.warning(f"  WARNING: {len(failed_tickers)} ticker(s) failed after all retries: {', '.join(failed_tickers)}")
    
    print(f"\n  Successfully downloaded {len(ticker_data)}/{len(tickers)} tickers")
    if logger:
        logger.info(f"  Successfully downloaded {len(ticker_data)}/{len(tickers)} tickers")
    
    return ticker_data

def main():
    # Load configuration
    try:
        start_date, end_date, stock_screener_file = load_config()
        print(f"Date range: {start_date} to {end_date}")
    except Exception as exc:
        print(f"Config error: {exc}")
        return 1
    
    # Load tickers
    try:
        tickers = load_tickers(stock_screener_file)
        print(f"Tickers: {', '.join(tickers)}")
    except Exception as exc:
        print(f"Ticker loading error: {exc}")
        return 1
    
    # Create output directory
    output_dir = Path("data")
    
    # Set up logger for ticker-level information
    log_file_path = output_dir / "update_data_log.txt"
    logger = setup_logger(log_file_path)
    logger.info("=" * 80)
    logger.info(f"Starting data update - Date range: {start_date} to {end_date}")
    logger.info(f"Tickers: {', '.join(tickers)}")
    
    # Download data for all tickers using bulk download
    print()
    ticker_data = download_ticker_data_bulk(tickers, start_date, end_date, logger=logger)
    
    if not ticker_data:
        print("ERROR: No data retrieved for any ticker")
        logger.error("ERROR: No data retrieved for any ticker")
        return 1
    
    # 1. Create dividends CSV (dates as rows, tickers as columns)
    # This ensures ALL dividends are captured, even when payment dates/frequencies differ
    print("\nCreating dividends CSV...")
    logger.info("\nCreating dividends CSV...")
    div_dict = {}
    div_counts = {}
    
    for ticker, df in ticker_data.items():
        # Extract all non-zero dividends (preserves all payment dates)
        div_series = df['Dividends'][df['Dividends'] > 0]
        if len(div_series) > 0:
            div_dict[ticker] = div_series
            div_counts[ticker] = len(div_series)
            logger.info(f"  {ticker}: {len(div_series)} dividend payments")
    
    if div_dict:
        # Create DataFrame: pandas will union all dates, ensuring no dividends are lost
        # NaN values will appear where a ticker didn't pay on a given date
        df_dividends = pd.DataFrame(div_dict)
        df_dividends.index.name = "date"
        df_dividends = df_dividends.reset_index()
        df_dividends["date"] = pd.to_datetime(df_dividends["date"]).dt.strftime("%Y-%m-%d")
        
        # Verify all dividends are included
        for ticker in div_dict.keys():
            non_null_count = df_dividends[ticker].notna().sum()
            if non_null_count != div_counts[ticker]:
                logger.warning(f"  WARNING: {ticker} expected {div_counts[ticker]} dividends, found {non_null_count} in DataFrame")
            else:
                logger.info(f"  ✓ {ticker}: All {div_counts[ticker]} dividends included")
        
        div_output_path = output_dir / "dividends.csv"
        df_dividends.to_csv(div_output_path, index=False)
        print(f"  Saved to {div_output_path}")
        print(f"  Shape: {df_dividends.shape[0]} rows × {df_dividends.shape[1]} columns")
    else:
        print("  WARNING: No dividend data found for any ticker")
        # Create empty CSV with date column
        df_dividends = pd.DataFrame(columns=["date"] + list(ticker_data.keys()))
        div_output_path = output_dir / "dividends.csv"
        df_dividends.to_csv(div_output_path, index=False)
        print(f"  Created empty dividends CSV at {div_output_path}")
    
    # 2. Create prices CSV (tickers as rows, start_price and end_price as columns)
    print("\nCreating prices CSV...")
    price_rows = []
    
    for ticker, df in ticker_data.items():
        if len(df) == 0:
            continue
        
        # Get first and last trading days in the range
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        
        price_rows.append({
            'ticker': ticker,
            'start_price': start_price,
            'end_price': end_price
        })
    
    if price_rows:
        df_prices = pd.DataFrame(price_rows)
        price_output_path = output_dir / "prices.csv"
        df_prices.to_csv(price_output_path, index=False)
        print(f"  Saved to {price_output_path}")
        print(f"  Shape: {df_prices.shape[0]} rows × {df_prices.shape[1]} columns")
        print(f"\nPrice summary:")
        logger.info("\nPrice summary:")
        for _, row in df_prices.iterrows():
            logger.info(f"  {row['ticker']}: ${row['start_price']:.2f} → ${row['end_price']:.2f}")
    else:
        print("  ERROR: No price data available")
        return 1
    
    print("\nDone!")
    logger.info("\nDone!")
    logger.info("=" * 80)
    return 0

if __name__ == "__main__":
    exit(main())
