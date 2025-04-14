import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# {'RELIANCE': 250, 'INFY': 1000, 'TCS': 300, ...} aise hai dictionary
from lotsize import lot_size_dict
folder_path = r'Options csv\cleaned_csvs'
output_csv = 'bulk_summary_output.csv'
output_html = 'options_analysis_report.html'
charts_folder = 'risk_profiles'
if not os.path.exists(charts_folder):
    os.makedirs(charts_folder)
summary_rows = []
trading_signals = []
all_stocks_data = {}

def calculate_max_pain(df, lot_size):
    """Calculate the max pain point (strike price where option writers have least liability)"""
    strikes = df['STRIKE'].unique()
    pain_values = []
    for strike in strikes:
        pain = sum(df[df['STRIKE'] < strike]['Put_OI.1'].fillna(0) * lot_size * (strike - df[df['STRIKE'] < strike]['STRIKE']))
        pain += sum(df[df['STRIKE'] > strike]['Call_OI'].fillna(0) * lot_size * (df[df['STRIKE'] > strike]['STRIKE'] - strike))
        pain_values.append((strike, pain))
    if pain_values:
        max_pain_point = min(pain_values, key=lambda x: x[1])[0]
        return max_pain_point
    return None

def calculate_atm_strike(df):
    """Estimate the at-the-money strike by finding strike with closest call/put premiums"""
    # absolute diff call and puts ka
    df['price_diff'] = abs(df['Call_LTP'] - df['Put_LTP.1'])
    atm_row = df.loc[df['price_diff'].idxmin()]
    return atm_row['STRIKE']

def calculate_implied_move(df, atm_strike):
    """Estimate the implied move based on straddle pricing"""
    atm_options = df[df['STRIKE'] == atm_strike]
    if not atm_options.empty:
        atm_call = atm_options['Call_LTP'].iloc[0]
        atm_put = atm_options['Put_LTP.1'].iloc[0]
        return atm_call + atm_put
    return None

def calculate_risk_score(df, oi_ratio, iv_skew=None):
    """Calculate a risk score (0-100) based on options data"""
    # 50 neutral score hai, start wahi se hota hai
    risk_score = 50
    if oi_ratio > 1.5:
        risk_score -= 15  # More calls toh bullish toh kam riks
    elif oi_ratio < 0.7:
        risk_score += 15  # More puts toh bearish toh bohot riks
    call_vol_oi_max = df['Call_VOL/OI'].max()
    put_vol_oi_max = df['Put_VOL/OI'].max()
    if call_vol_oi_max > 5:
        risk_score -= 10  # Heavy call buying toh bullish signal
    if put_vol_oi_max > 5:
        risk_score += 10  # Heavy put buying toh bearish signal
    # Adjust according to concentration of OI
    top_5_call_oi = df.nlargest(5, 'Call_OI')['Call_OI'].sum()
    top_5_put_oi = df.nlargest(5, 'Put_OI.1')['Put_OI.1'].sum()
    total_call_oi = df['Call_OI'].sum()
    total_put_oi = df['Put_OI.1'].sum()
    if total_call_oi > 0 and top_5_call_oi / total_call_oi > 0.7:
        risk_score += 5  # Concentrated call OI toh jyada riks
    if total_put_oi > 0 and top_5_put_oi / total_put_oi > 0.7:
        risk_score += 5  # Concentrated put OI toh jyada riks
    return max(0, min(100, risk_score))

def generate_trading_signal(row):
    """Generate trading signals based on option data analysis"""
    signal = "HOLD"
    confidence = "Medium"
    strategy = "N/A"
    # Convert OI Ratio to numeric
    oi_ratio = row['OI Ratio']
    if oi_ratio == 'âˆž':
        oi_ratio = float('inf')
    else:
        oi_ratio = float(oi_ratio)
    risk_score = row['Risk Score']
    put_call_ratio = row.get('Put/Call Volume Ratio', 1.0)
    # Trading signals logic 
    if oi_ratio > 1.5 and risk_score < 40:
        signal = "BUY"
        confidence = "High" if oi_ratio > 2.0 else "Medium"
        strategy = "Bull Call Spread"
    elif oi_ratio < 0.6 and risk_score > 60:
        signal = "SELL"
        confidence = "High" if oi_ratio < 0.4 else "Medium"
        strategy = "Bear Put Spread"
    elif 0.8 <= oi_ratio <= 1.2 and risk_score >= 45 and risk_score <= 55:
        signal = "NEUTRAL"
        strategy = "Iron Condor"
    elif row.get('IV Percentile', 0) > 80:
        signal = "SELL VOLATILITY"
        strategy = "Short Straddle/Strangle"
    elif row.get('IV Percentile', 0) < 20:
        signal = "BUY VOLATILITY"
        strategy = "Long Straddle/Strangle"
    return {
        'Stock': row['Stock'],
        'Signal': signal,
        'Confidence': confidence,
        'Strategy': strategy
    }
def create_risk_profile(symbol, df, charts_folder):
    try:
        # Create a figure with a single row of 2 plots
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        
        # oi distribution (Chart 1: Left)
        sns.barplot(x='STRIKE', y='Call_OI', data=df, color='green', alpha=0.6, label='Calls', ax=ax1)
        put_data = df.copy()
        put_data['Put_OI.1'] = -put_data['Put_OI.1']
        sns.barplot(x='STRIKE', y='Put_OI.1', data=put_data, color='red', alpha=0.6, label='Puts', ax=ax1)
        ax1.set_title(f'{symbol} - Open Interest Distribution', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45, labelsize=7)
        ax1.tick_params(axis='y', labelsize=7)

        # Trading volume distribution (Chart 2: Right)
        ax2.bar(df['STRIKE'] - 0.5, df['Call_VOLUME'], width=1, color='green', alpha=0.6, label='Call Volume')
        ax2.bar(df['STRIKE'] + 0.5, df['Put_VOLUME.1'], width=1, color='red', alpha=0.6, label='Put Volume')
        ax2.set_title(f'{symbol} - Trading Volume Distribution', fontsize=10)
        ax2.set_xlabel('Strike Price', fontsize=8)
        ax2.set_ylabel('Volume', fontsize=8)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45, labelsize=7)
        ax2.tick_params(axis='y', labelsize=7)
    
        plt.tight_layout(pad=2.5)
        
        chart_path = os.path.join(charts_folder, f'{symbol}_risk_profile.png')
        plt.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close(fig)  # Explicitly close the figure to address the warning
        
        return chart_path
    except Exception as e:
        print(f"Could not create chart for {symbol}: {e}")
        return None

def generate_html_report(summary_df, signals_df, charts_folder):
    """Generate a detailed HTML report of the options analysis"""
    # Count of stocks by sentiment
    sentiment_counts = summary_df['Sentiment'].value_counts()
    bullish_count = sentiment_counts.get('Bullish', 0)
    bearish_count = sentiment_counts.get('Bearish', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    # Signal those counts
    signal_counts = signals_df['Signal'].value_counts()
    buy_count = signal_counts.get('BUY', 0)
    sell_count = signal_counts.get('SELL', 0)
    hold_count = signal_counts.get('HOLD', 0)
    neutral_signal = signal_counts.get('NEUTRAL', 0)
    # 5 highest and lowest riks stocks bana raha hun
    if 'Risk Score' in summary_df.columns:
        top_risky = summary_df.nlargest(5, 'Risk Score')[['Stock', 'Risk Score']]
        top_safe = summary_df.nsmallest(5, 'Risk Score')[['Stock', 'Risk Score']]
        risky_stocks_html = ""
        for _, row in top_risky.iterrows():
            risky_stocks_html += f"""
            <tr>
                <td>{row['Stock']}</td>
                <td>{row['Risk Score']}</td>
            </tr>
            """
        safe_stocks_html = ""
        for _, row in top_safe.iterrows():
            safe_stocks_html += f"""
            <tr>
                <td>{row['Stock']}</td>
                <td>{row['Risk Score']}</td>
            </tr>
            """
    else:
        risky_stocks_html = "<tr><td colspan='2'>Risk Score not available</td></tr>"
        safe_stocks_html = "<tr><td colspan='2'>Risk Score not available</td></tr>"
    # signals table
    signals_html = ""
    for _, row in signals_df.iterrows():
        signal_color = "text-success" if row['Signal'] == 'BUY' else "text-danger" if row['Signal'] == 'SELL' else "text-warning"
        signals_html += f"""
        <tr>
            <td>{row['Stock']}</td>
            <td class="{signal_color} fw-bold">{row['Signal']}</td>
            <td>{row['Confidence']}</td>
            <td>{row['Strategy']}</td>
        </tr>
        """
    # summary table
    summary_html = ""
    for _, row in summary_df.iterrows():
        sentiment_color = "text-success" if row['Sentiment'] == 'Bullish' else "text-danger" if row['Sentiment'] == 'Bearish' else "text-warning"
        chart_path = f"{charts_folder}/{row['Stock']}_risk_profile.png"
        # Check if chart exists
        chart_link = f'<a href="{chart_path}" target="_blank">View Chart</a>' if os.path.exists(chart_path) else 'Not Available'
        summary_html += f"""
        <tr>
            <td>{row['Stock']}</td>
            <td>{row['Call Capital']:,}</td>
            <td>{row['Put Capital']:,}</td>
            <td>{row['OI Ratio']}</td>
            <td class="{sentiment_color}">{row['Sentiment']}</td>
            <td>{row.get('Risk Score', 'N/A')}</td>
            <td>{row.get('Max Pain', 'N/A')}</td>
            <td>{chart_link}</td>
        </tr>
        """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Options Market Analysis - {datetime.now().strftime('%Y-%m-%d')}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
        <link href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css" rel="stylesheet">
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

        <style>
            body {{
                background-color: #f9f9f9;
                padding-top: 70px;
            }}
            .card {{
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            }}
            .card-header {{
                background-color: #e9ecef;
                font-weight: bold;
            }}
            .table-responsive {{
                max-height: 600px;
                overflow-y: auto;
            }}
            .navbar-brand {{
                color: #007bff !important;
            }}
        </style>
    </head>
    <body>
        <!-- Floating Navbar -->
        <nav class="navbar navbar-expand-lg navbar-light bg-light shadow-sm fixed-top">
            <div class="container-fluid">
                <a class="navbar-brand fw-bold" href="#">Options Dashboard</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav ms-auto">
                        <li class="nav-item"><a class="nav-link" href="#marketSentimentChart">Market Sentiment</a></li>
                        <li class="nav-item"><a class="nav-link" href="#tradingSignalsChart">Trading Signals</a></li>
                        <li class="nav-item"><a class="nav-link" href="#risktable">Risk Tables</a></li>
                        <li class="nav-item"><a class="nav-link" href="#traderec">Trading Recommendations</a></li>
                        <li class="nav-item"><a class="nav-link" href="#summarytable">Summary</a></li>
                    </ul>
                </div>
            </div>
        </nav>

        <div class="container-fluid mt-4">
            <div class="row mb-4">
                <div class="col-12">
                    <h1 class="text-center">Options Market Analysis</h1>
                </div>
            </div>
            
            <!*-* Market Overview *-*>
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Market Sentiment</div>
                        <div class="card-body">
                            <canvas id="marketSentimentChart" height="100"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Trading Signals</div>
                        <div class="card-body">
                            <canvas id="tradingSignalsChart" height="100"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header" id ="risktable">Top Risky Stocks</div>
                        <div class="card-body">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Stock</th>
                                        <th>Risk Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {risky_stocks_html}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Top Safe Stocks</div>
                        <div class="card-body">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Stock</th>
                                        <th>Risk Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {safe_stocks_html}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Trading Signals -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header" id ="traderec">Trading Recommendations</div>
                        <div class="card-body table-responsive">
                            <table class="table table-striped table-hover">
                                <thead>
                                    <tr>
                                        <th>Stock</th>
                                        <th>Signal</th>
                                        <th>Confidence</th>
                                        <th>Recommended Strategy</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {signals_html}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <!*-* Options Analysis Summary *-*>
            <div class="row">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header" id = "summarytable">Options Analysis Summary</div>
                        <div class="card-body table-responsive">
                            <table class="table table-striped table-hover">
                                <thead>
                                    <tr>
                                        <th>Stock</th>
                                        <th>Call Capital</th>
                                        <th>Put Capital</th>
                                        <th>OI Ratio</th>
                                        <th>Sentiment</th>
                                        <th>Risk Score</th>
                                        <th>Max Pain</th>
                                        <th>Risk Chart</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {summary_html}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
    // Market sentiment chart
    var sentimentCtx = document.getElementById('marketSentimentChart').getContext('2d');
    var sentimentChart = new Chart(sentimentCtx, {{
        type: 'pie',
        data: {{
            labels: ['Bullish', 'Bearish', 'Neutral'],
            datasets: [{{
                data: [{bullish_count}, {bearish_count}, {neutral_count}],
                backgroundColor: [
                    'rgba(0, 255, 0, 1)',  // Green
                    'rgba(255,0, 0, 1)', // Red
                    'rgba(255, 235, 59, 0.7)' // Yellow
                ]
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{
                legend: {{
                    position: 'bottom',
                }},
                title: {{
                    display: true,
                    text: 'Market Sentiment Distribution'
                }}
            }}
        }}
    }});

    // Trading signals chart
    var signalsCtx = document.getElementById('tradingSignalsChart').getContext('2d');
    var signalsChart = new Chart(signalsCtx, {{
        type: 'pie',
        data: {{
            labels: ['Buy', 'Sell', 'Hold', 'Neutral'],
            datasets: [{{
                data: [{buy_count}, {sell_count}, {hold_count}, {neutral_signal}],
                backgroundColor: [
                    'rgba(0, 255, 0, 1)',  // Green
                    'rgba(255, 0, 0, 1)', // Red
                    'rgba(255, 235, 59, 0.7)',// Yellow
                    'rgba(144, 164, 174, 0.7)'// Gray for Neutral
                ]
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{
                legend: {{
                    position: 'bottom',
                }},
                title: {{
                    display: true,
                    text: 'Trading Signals Distribution'
                }}
            }}
        }}
    }});
</script>
    </body>
    </html>
    """
    with open(output_html, 'w') as f:
        f.write(html)
    return output_html

# **** MAIN PROCESSING FUNCTION ****
def process_options_data(folder_path, lot_size_dict, charts_folder):
    """Process all option chain files in the folder"""
    summary_rows = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            filename_parts = filename.split('-')
            symbol = filename_parts[3] if len(filename_parts) > 3 else 'UNKNOWN'
            lot_size = lot_size_dict.get(symbol.upper(), 1000)
            try:
                df = pd.read_csv(file_path)
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])
                cols_to_numeric = ['Call_LTP', 'Call_OI', 'Call_VOLUME', 'Put_LTP.1', 'Put_OI.1', 'Put_VOLUME.1', 'STRIKE']
                
                if 'Call_IV' in df.columns and 'Put_IV.1' in df.columns:
                    cols_to_numeric.extend(['Call_IV', 'Put_IV.1'])
                df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
                
                df['Call_VOL/OI'] = df['Call_VOLUME'] / df['Call_OI'].replace(0, np.nan)
                df['Put_VOL/OI'] = df['Put_VOLUME.1'] / df['Put_OI.1'].replace(0, np.nan)
                unusual_calls = df[df['Call_VOL/OI'] > 2.0]
                unusual_puts = df[df['Put_VOL/OI'] > 2.0]
                # ATM strike and implied move
                try:
                    atm_strike = calculate_atm_strike(df)
                    implied_move = calculate_implied_move(df, atm_strike)
                except Exception as e:
                    print(f"Could not calculate ATM metrics for {symbol}: {e}")
                    atm_strike = None
                    implied_move = None
                # Capital calcs
                df['Call_CAPITAL'] = df['Call_OI'] * lot_size * df['Call_LTP']
                df['Put_CAPITAL'] = df['Put_OI.1'] * lot_size * df['Put_LTP.1']
                total_call_cap = df['Call_CAPITAL'].sum()
                total_put_cap = df['Put_CAPITAL'].sum()
                # OI Sentiment
                call_oi_total = df['Call_OI'].sum()
                put_oi_total = df['Put_OI.1'].sum()
                oi_ratio = round(call_oi_total / put_oi_total, 2) if put_oi_total != 0 else float('inf')
                sentiment = "Bullish" if oi_ratio > 1.2 else "Bearish" if oi_ratio < 0.8 else "Neutral"
                # Advanced metrics
                max_pain = calculate_max_pain(df, lot_size)
                # IV Skew
                iv_skew = None
                if 'Call_IV' in df.columns and 'Put_IV.1' in df.columns:
                    # Only calculate with valid IV data
                    valid_iv = df.dropna(subset=['Call_IV', 'Put_IV.1'])
                    if not valid_iv.empty:
                        iv_skew = (valid_iv['Call_IV'] - valid_iv['Put_IV.1']).mean()
                # Volume ratios
                total_call_vol = df['Call_VOLUME'].sum()
                total_put_vol = df['Put_VOLUME.1'].sum()
                volume_ratio = round(total_put_vol / total_call_vol, 2) if total_call_vol != 0 else float('inf')
                # Risk score
                risk_score = calculate_risk_score(df, oi_ratio, iv_skew)
                # Generate risk profile chart
                chart_path = create_risk_profile(symbol, df, charts_folder)
                # Store processed dataframe for later use
                all_stocks_data[symbol] = df
                summary_row = {
                    'Stock': symbol,
                    'Call Capital': int(total_call_cap),
                    'Put Capital': int(total_put_cap),
                    'OI Ratio': round(oi_ratio, 2) if put_oi_total != 0 else 'âˆž',
                    'Sentiment': sentiment,
                    'Unusual CALLs': len(unusual_calls),
                    'Unusual PUTs': len(unusual_puts),
                    'Risk Score': risk_score,
                    'Max Pain': max_pain,
                    'Put/Call Volume Ratio': volume_ratio,
                    'ATM Strike': atm_strike,
                    'Implied Move': implied_move
                }
                if iv_skew is not None:
                    summary_row['IV Skew'] = round(iv_skew, 2)
                summary_rows.append(summary_row)
            except Exception as e:
                print(f"Failed on {filename}: {e}")
    return summary_rows

if __name__ == "__main__":
    summary_rows = process_options_data(folder_path, lot_size_dict, charts_folder)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if len(summary_df) >= 3:  # min 3 sample req for meaningful clustering
            try:
                numeric_columns = ['Call Capital', 'Put Capital', 'Risk Score']
                # Handling special oi ratio values
                summary_df['OI Ratio Numeric'] = summary_df['OI Ratio'].apply(
                    lambda x: float('inf') if x == 'âˆž' else float(x))
                features = summary_df[numeric_columns + ['OI Ratio Numeric']].copy()
                features.replace([float('inf'), -float('inf')], 9999, inplace=True)
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                # K-means clustering
                n_clusters = min(3, len(summary_df))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                summary_df['Market_Segment'] = kmeans.fit_predict(scaled_features)
                # giving meaning to clusters
                segments = {
                    0: "High Volatility",
                    1: "Bullish Momentum",
                    2: "Bearish Pressure"
                }
                summary_df['Market_Segment'] = summary_df['Market_Segment'].map(
                    lambda x: segments.get(x, "Other")
                )
            except Exception as e:
                print(f"Could not add market segmentation: {e}")
        for _, row in summary_df.iterrows():
            signal = generate_trading_signal(row)
            trading_signals.append(signal)
        signals_df = pd.DataFrame(trading_signals)
        summary_df.to_csv(output_csv, index=False)
        signals_df.to_csv('trading_signals.csv', index=False)
        html_report = generate_html_report(summary_df, signals_df, charts_folder)
        print(f"Analysis complete. Summary saved to '{output_csv}' with {len(summary_df)} entries.")
        print(f"Trading signals saved to 'trading_signals.csv' with {len(signals_df)} entries.")
        print(f"HTML report generated at '{html_report}'")
    else:
        print("No data was processed successfully.")