from pathlib import Path
import logging
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_prices(data_path="data/prices.csv"):
    df = pd.read_csv(data_path)
    
    # Ensure index is ticker and is uppercase
    df["ticker"] = df["ticker"].str.upper()
    
    # Set index to ticker
    df = df.set_index("ticker")
    
    return df

def load_stock_screener_data(data_path):
    if not Path(data_path).exists():
        raise FileNotFoundError(f"{data_path} not found")
    
    df = pd.read_csv(data_path)
    df = df[["Ticker", "Sector", "Fwd. Yield", "Div. 1Y Chg (%)", "Div. 3Y Avg (%)",
        "Div. 5Y Avg (%)", "Div. 10Y Avg (%)"]]
    df.columns = ["ticker", "sector", "div_yield", "div_cagr_1y", "div_cagr_3y", "div_cagr_5y", "div_cagr_10y"]

    # Convert percentage strings to numeric values
    for col in ["div_yield", "div_cagr_1y", "div_cagr_3y", "div_cagr_5y", "div_cagr_10y"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    # If there's a row where the ticker is "Summary", remove it.
    df = df[df["ticker"].str.upper() != "SUMMARY"]

    return df

def blend_div_cagrs(div_cagrs_array, weights):
    return np.dot(div_cagrs_array, weights)

def compute_nav_cagrs(prices_df):
    cagrs = {}
    for ticker, row in prices_df.iterrows():
        price_then = row["start_price"]
        price_now = row["end_price"]

        if price_then <= 0 or price_now <= 0 or np.isnan(price_then) or np.isnan(price_now):
            raise ValueError(f"No valid price data for {ticker}")
        
        nominal_cagr = (price_now / price_then) ** (1 / 5) - 1
        cagrs[ticker] = nominal_cagr

    return cagrs

def calculate_portfolio_allocation(weights, tickers, available_cash, prices_df):
    results = []
    
    for i, ticker in enumerate(tickers):
        weight = weights[i]
        ticker_upper = ticker.upper()
        
        # Get current price (end_price from prices_df)
        if ticker_upper not in prices_df.index:
            raise ValueError(f"Ticker {ticker_upper} not found in prices_df")
        
        current_price = prices_df.loc[ticker_upper, "end_price"]
        
        if current_price <= 0 or np.isnan(current_price):
            raise ValueError(f"Invalid price for {ticker_upper}: {current_price}")
        
        # Calculate dollar allocation
        allocation_dollars = weight * available_cash
        
        # Calculate shares (round down to whole shares)
        shares = int(allocation_dollars / current_price)
        
        # Calculate actual cost (shares * price)
        cost = shares * current_price
        
        results.append({
            'ticker': ticker_upper,
            'weight': weight,
            'allocation_dollars': allocation_dollars,
            'shares': shares,
            'cost': cost
        })
    
    df = pd.DataFrame(results)
    return df

def portfolio_div_income_cagr(weights, div_yields, div_cagrs, horizon=5):
    # Current annual dividend income (scalar)
    current_income = np.dot(weights, div_yields)
    if current_income <= 0:
        return 0.0
        
    # Income in `horizon` years: each asset's yield grows at its own div_cagr
    growth_factors = np.power(1 + div_cagrs, horizon)
    future_income = np.dot(weights, div_yields * growth_factors)

    cagr = (future_income / current_income) ** (1 / horizon) - 1
    return cagr

def objective_max_div_cagr_portfolio(w, div_yields_array, div_cagrs_array, lambda_diversity):
    w = np.asarray(w)
    port_div_cagr = portfolio_div_income_cagr(w, div_yields_array, div_cagrs_array)
    concentration_penalty = lambda_diversity * np.sum(w**2)  # HHI term
    return -port_div_cagr + concentration_penalty

def generate_random_initial_guess(n, weight_bounds):
    # Generate random weights within bounds
    w = np.zeros(n)
    for i, (low, high) in enumerate(weight_bounds):
        w[i] = np.random.uniform(low, high)
    
    # Normalize to sum to 1
    total = np.sum(w)
    if total > 0:
        w = w / total
    else:
        # Fallback to equal weights if all zeros
        w = np.ones(n) / n
    
    return w

def optimize_for_target_yield(target_yield, div_cagrs_array, nav_cagrs_array, div_yields_array, 
                               tickers, weight_bounds, min_real_nav_growth, inflation_rate,
                               lambda_diversity, bucket_indices, bucket_total_ranges,
                               sector_indices, sector_max_weights, num_random_starts):
    n = len(tickers)
    
    # Constraints
    constraints = []
    
    # 1. Weights sum to 1
    constraints.append({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1.0
    })
    
    # 2. Portfolio yield equals target yield
    constraints.append({
        'type': 'eq',
        'fun': lambda w: np.dot(w, div_yields_array) - target_yield
    })

    # 3. Real NAV growth constraint
    nav_cagr_for_opt = nav_cagrs_array.copy()
    target_nav_cagr = min_real_nav_growth + inflation_rate
    constraints.append({
        'type': 'ineq',
        'fun': lambda w: np.dot(w, nav_cagr_for_opt) - target_nav_cagr
    })
    
    # 4. Bucket total_range constraints
    for bucket_indices, total_range in zip(bucket_indices, bucket_total_ranges):
        if len(bucket_indices) > 0:
            # Lower bound: sum(weights[bucket]) >= total_range[0]
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idxs=bucket_indices, min_val=total_range[0]: np.sum(w[idxs]) - min_val
            })
            # Upper bound: sum(weights[bucket]) <= total_range[1]
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idxs=bucket_indices, max_val=total_range[1]: max_val - np.sum(w[idxs])
            })

    # 5. Sector max weight constraints
    for sector_idx_list, max_w in zip(sector_indices, sector_max_weights):
        if len(sector_idx_list) > 0:
            # Upper bound: sum(weights[sector]) <= max_weight
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idxs=sector_idx_list, max_val=max_w: max_val - np.sum(w[idxs])
            })
    
    # Random search over initial guesses
    best_result = None
    best_objective = -np.inf
    best_initial_guess_idx = None
    successful_runs = 0
    
    for i in range(num_random_starts):
        # Generate random initial guess
        if i == 0:
            # First guess: equal weights
            x0 = np.ones(n) / n
        else:
            # Subsequent guesses: random within bounds
            x0 = generate_random_initial_guess(n, weight_bounds)
        
        # Optimize
        result = minimize(
            objective_max_div_cagr_portfolio,
            x0,
            args=(div_yields_array, div_cagrs_array, lambda_diversity),
            method='SLSQP',
            bounds=weight_bounds,   
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            successful_runs += 1
            weights = result.x
            # Calculate objective value (negative because we minimize negative div CAGR)
            objective_value = -result.fun
            
            # Track best solution
            if objective_value > best_objective:
                best_objective = objective_value
                best_initial_guess_idx = i
                best_result = result
    
    # If no successful runs, return failure
    if best_result is None:
        return {
            'success': False,
            'error_message': f'All {num_random_starts} random starts failed',
            'target_yield': target_yield
        }
    
    # Process best result
    weights = best_result.x
    port_div_cagr = portfolio_div_income_cagr(weights, div_yields_array, div_cagrs_array)
    port_nav_cagr = np.dot(weights, nav_cagrs_array)
    port_div_yield = np.dot(weights, div_yields_array)
    real_nav_growth = port_nav_cagr - inflation_rate
    
    # Count number of invested assets (weights above threshold)
    num_invested_assets = np.sum(weights > 1e-6)
    
    return {
        'weights': weights,
        'div_cagr': port_div_cagr,
        'div_yield': port_div_yield,
        'nav_cagr': port_nav_cagr,
        'real_nav_growth': real_nav_growth,
        'num_invested_assets': num_invested_assets,
        'success': True,
        'random_search_stats': {
            'total_starts': num_random_starts,
            'successful_starts': successful_runs,
            'best_initial_guess_idx': best_initial_guess_idx
        }
    }

def main():
    # Load configuration
    try:
        config = load_config()
        inflation_rate = config["inputs"]["inflation_rate"]
        min_real_nav_growth = config["inputs"]["min_real_nav_growth_rate"]
        lambda_diversity = config["inputs"]["lambda_diversity"]
        blended_div_cagrs_weights = config["inputs"]["blended_div_cagrs_weights"]
        print("Config loaded successfully")
    except Exception as exc:
        print(f"Config error: {exc}")
        return 1
    
    # Set random seed for reproducibility
    np.random.seed(config["inputs"]["rng_seed"])
    
    # Load stock screener data
    print("\nLoading stock screener data...")
    try:
        stock_screener_df = load_stock_screener_data(config["inputs"]["stock_screener_file"])
        tickers = stock_screener_df['ticker'].values
        div_yields_array = stock_screener_df['div_yield'].values
        print(f"  Loaded {len(tickers)} tickers")

        # Blend the div CAGRs
        all_div_cagrs_array = stock_screener_df[['div_cagr_1y', 'div_cagr_3y', 'div_cagr_5y', 'div_cagr_10y']].values
        div_cagrs_array = blend_div_cagrs(all_div_cagrs_array, blended_div_cagrs_weights)
        
        # Check that all tickers have a sector
        all_sectors = set()
        for bucket, params in config['buckets'].items():
            all_sectors.update(params['sectors'])
        
        tickers_without_sector = []
        for ticker in tickers:
            sector = stock_screener_df.loc[stock_screener_df['ticker'] == ticker, 'sector'].values[0]
            if pd.isna(sector) or sector not in all_sectors:
                tickers_without_sector.append(ticker)
        
        if tickers_without_sector:
            print(f"\n✗ ERROR: Found {len(tickers_without_sector)} ticker(s) without a valid sector:")
            for ticker in tickers_without_sector:
                sector = stock_screener_df.loc[stock_screener_df['ticker'] == ticker, 'sector'].values[0]
                print(f"  {ticker}: sector = {sector}")
            return 1
        
        # Categorize tickers by sector and build bucket indices
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        sector_bucket_indices = []
        bucket_total_ranges = []
        bucket_names = []
        for bucket, params in config['buckets'].items():
            bucket_tickers = stock_screener_df[
                stock_screener_df['sector'].isin(params['sectors'])
            ]['ticker'].tolist()
            indices = [ticker_to_idx[t] for t in bucket_tickers if t in ticker_to_idx]
            sector_bucket_indices.append(indices)
            bucket_total_ranges.append(params['total_range'])
            bucket_names.append(bucket)
        
        # Print bucket stock counts
        print("\nBucket stock counts:")
        for bucket_name, indices, total_range in zip(bucket_names, sector_bucket_indices, bucket_total_ranges):
            print(f"  {bucket_name}: {len(indices)} stocks (total_range: {total_range[0]:.0%} - {total_range[1]:.0%})")
        
        # Build sector indices and max weights
        sector_indices = []
        sector_max_weights = []
        sector_names = []
        for sector_name, sector_params in config['sectors'].items():
            sector_tickers = stock_screener_df[
                stock_screener_df['sector'] == sector_name
            ]['ticker'].tolist()
            indices = [ticker_to_idx[t] for t in sector_tickers if t in ticker_to_idx]
            if len(indices) > 0:
                sector_indices.append(indices)
                sector_max_weights.append(sector_params['max_weight'])
                sector_names.append(sector_name)
        
        # Print sector stock counts
        print("\nSector stock counts:")
        for sector_name, indices, max_weight in zip(sector_names, sector_indices, sector_max_weights):
            print(f"  {sector_name}: {len(indices)} stocks (max_weight: {max_weight:.0%})")

        # Portfolio weight bounds
        weight_bounds = []
        for ticker in tickers:
            sector = stock_screener_df.loc[stock_screener_df['ticker'] == ticker, 'sector'].values[0]
            found = False
            for bucket, params in config['buckets'].items():
                if sector in params['sectors']:
                    weight_bounds.append(tuple(params['per_stock_range']))
                    found = True
                    break
            if not found:
                # This shouldn't happen due to check above, but just in case
                print(f"  WARNING: {ticker} sector {sector} not in any bucket, using default bounds")
                weight_bounds.append((0.0, 1.0))
    except Exception as exc:
        print(f"Stock screener data loading error: {exc}")
        return 1
    
    # Load prices
    print("\nLoading price data...")
    try:
        prices_df = load_prices()
        print(f"  Loaded prices for {len(prices_df)} tickers")
    except Exception as exc:
        print(f"Prices loading error: {exc}")
        return 1

    # Compute NAV CAGRs
    print("\nComputing NAV CAGRs...")
    try:
        nav_cagrs = compute_nav_cagrs(prices_df)

        # Convert to array in ticker order
        nav_cagrs_array = np.array([nav_cagrs.get(t, 0.0) for t in tickers])
        print(f"  NAV CAGRs computed for {len(nav_cagrs)} tickers")
        print(f"  NAV CAGR range: {nav_cagrs_array.min():.2%} to {nav_cagrs_array.max():.2%}")
    except Exception as exc:
        print(f"NAV CAGR computation error: {exc}")
        return 1
    
    # Get target yield and random search points from config
    target_yield = config['inputs']['target_yield']
    num_random_starts = config['inputs']['opt_points']
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION STARTING")
    print(f"{'='*80}")
    print(f"Objective: Maximize dividend CAGR given target yield")
    print(f"Target yield: {target_yield:.2%}")
    print(f"Minimum real NAV growth: {min_real_nav_growth:.2%}")
    print(f"Inflation rate: {inflation_rate:.2%}")
    print(f"Number of assets: {len(tickers)}")
    print(f"Lambda diversity: {lambda_diversity}")
    print(f"Random initial guesses: {num_random_starts}")
    print(f"{'='*80}\n")
    
    # Run optimization
    print(f"Optimizing for target yield: {target_yield:.2%} ({num_random_starts} random starts)...", end=" ", flush=True)
    
    result = optimize_for_target_yield(
        target_yield, 
        div_cagrs_array, 
        nav_cagrs_array, 
        div_yields_array,
        tickers, 
        weight_bounds,
        min_real_nav_growth, 
        inflation_rate,
        lambda_diversity,
        sector_bucket_indices,
        bucket_total_ranges,
        sector_indices,
        sector_max_weights,
        num_random_starts
    )
    
    if result and result.get('success', False):
        print(f"✓ Success")
        
        # Print random search stats if available
        if 'random_search_stats' in result:
            stats = result['random_search_stats']
            print(f"  (Best from initial guess #{stats['best_initial_guess_idx']+1}, {stats['successful_starts']}/{stats['total_starts']} successful)")
        
        # Print results
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION RESULTS")
        print(f"{'='*80}")
        print(f"Target yield: {target_yield:.2%}")
        print(f"Portfolio dividend yield: {result['div_yield']:.2%}")
        print(f"Portfolio dividend CAGR: {result['div_cagr']:.2%}")
        print(f"Portfolio NAV CAGR: {result['nav_cagr']:.2%}")
        print(f"Real portfolio NAV CAGR: {result['real_nav_growth']:.2%}")
        print(f"Number of invested assets: {result['num_invested_assets']}")
        if 'random_search_stats' in result:
            stats = result['random_search_stats']
            print(f"\nRandom search: {stats['successful_starts']}/{stats['total_starts']} successful, best from guess #{stats['best_initial_guess_idx']+1}")
        print(f"{'='*80}\n")
        
        # Calculate bucket allocations
        weights = result['weights']
        epsilon = 1e-6
        
        print("Portfolio Allocation by Bucket:")
        print(f"{'Bucket':<20} {'Weight':<12} {'Assets':<10} {'Target Range':<20}")
        print("-" * 65)
        for bucket_name, bucket_indices, total_range in zip(bucket_names, sector_bucket_indices, bucket_total_ranges):
            if len(bucket_indices) > 0:
                bucket_weight = np.sum(weights[bucket_indices])
                bucket_assets = np.sum(weights[bucket_indices] > epsilon)
                print(f"{bucket_name:<20} {bucket_weight:>10.2%} {bucket_assets:>8} {total_range[0]:>8.0%} - {total_range[1]:>8.0%}")
        print()
        
        # Calculate sector allocations
        print("Portfolio Allocation by Sector:")
        print(f"{'Sector':<25} {'Weight':<12} {'Assets':<10}")
        print("-" * 50)
        sector_weights = {}
        sector_counts = {}
        for i, ticker in enumerate(tickers):
            sector = stock_screener_df.loc[stock_screener_df['ticker'] == ticker, 'sector'].values[0]
            if sector not in sector_weights:
                sector_weights[sector] = 0.0
                sector_counts[sector] = 0
            sector_weights[sector] += weights[i]
            if weights[i] > epsilon:
                sector_counts[sector] += 1
        
        # Sort sectors by weight descending
        sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
        for sector, weight in sorted_sectors:
            count = sector_counts[sector]
            print(f"{sector:<25} {weight:>10.2%} {count:>8}")
        print()
        
        # Calculate portfolio allocation (shares and costs)
        available_cash = config['inputs']['available_cash']
        allocation_df = calculate_portfolio_allocation(weights, tickers, available_cash, prices_df)
        
        # Filter to only show tickers with non-zero weights
        allocation_df = allocation_df[allocation_df['weight'] > epsilon].copy()
        allocation_df = allocation_df.sort_values('weight', ascending=False)
        
        # Add current price column for reference
        allocation_df['current_price'] = allocation_df['ticker'].apply(
            lambda t: prices_df.loc[t, 'end_price']
        )
        
        # Reorder columns for better readability
        allocation_df = allocation_df[['ticker', 'current_price', 'weight', 'allocation_dollars', 'shares', 'cost']]
        
        print("Portfolio Allocation Details:")
        print(f"{'Ticker':<10} {'Weight':<12} {'Allocation':<15} {'Shares':<10} {'Cost':<15}")
        print("-" * 70)
        total_cost = 0
        for _, row in allocation_df.iterrows():
            print(f"{row['ticker']:<10} {row['weight']:>10.2%} ${row['allocation_dollars']:>13,.2f} {row['shares']:>9} ${row['cost']:>13,.2f}")
            total_cost += row['cost']
        print("-" * 70)
        print(f"{'TOTAL':<10} {'100.00%':>10} ${available_cash:>13,.2f} {'':>9} ${total_cost:>13,.2f}")
        remaining_cash = available_cash - total_cost
        print(f"{'Remaining':<10} {'':>10} {'':>15} {'':>9} ${remaining_cash:>13,.2f}")
        print()
        
        # Save allocation to CSV for trade execution
        allocation_output_path = Path("output") / "portfolio_allocation.csv"
        allocation_output_path.parent.mkdir(exist_ok=True)
        allocation_df.to_csv(allocation_output_path, index=False)
        print(f"✓ Saved portfolio allocation to {allocation_output_path}")
        print()
    else:
        error_msg = result.get('error_message', 'Unknown error') if result else 'Optimization returned None'
        print(f"✗ Failed: {error_msg}")
        return 1

    return 0
    
if __name__ == "__main__":
    exit(main())
