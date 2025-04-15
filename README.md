# Options Screener

A Python-based tool that analyzes and filters stock options data from CSV files, generating a comprehensive HTML report.

## 📂 Project Structure

- `downloadcsv.py`: Handles downloading of raw options data.
- `cleancsv.py`: Cleans and preprocesses the downloaded data.
- `lotsize.py`: Calculates lot sizes for options contracts.
- `nse.py`: Contains functions specific to NSE data handling.
- `risk_profiles/`: Directory containing risk assessment profiles.
- `sample_data/`: Contains sample CSV files for testing.
- `options_analysis_report.html`: Generated HTML report summarizing analysis.
- `bulk_summary_output.csv`, `trading_signals.csv`: CSV files with summarized analysis results.

## 🔧 Features

- **Data Acquisition**: Downloads raw options data for analysis.
- **Data Cleaning**: Processes and cleans the raw data for consistency.
- **Lot Size Calculation**: Computes lot sizes for each options contract.
- **Risk Assessment**: Evaluates risk profiles to filter options.
- **Report Generation**: Produces an HTML report summarizing the analysis.
- **Export Results**: Saves filtered call and put options into CSV and Excel files.

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rdmaR-05/Options-Screener.git
   cd Options-Screener
Install dependencies:

pip install -r requirements.txt
Prepare data:

Place your raw options CSV files into the sample_data/ directory.

Run the main script:

python main.py
View results:

Open options_analysis_report.html in your browser to view the analysis.


🛠 Technologies Used
-Python

-pandas

-numpy

-matplotlib

-seaborn

-openpyxl
