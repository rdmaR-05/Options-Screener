import os
import pandas as pd
input_folder = r"E:\Stocks\Options csv" 
output_folder = os.path.join(input_folder, "cleaned_csvs")
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(input_folder, file)
        # Read CSV again, now using the 2nd row as the header
        df = pd.read_csv(file_path, skiprows=1, header=0)
        # Strip spaces from column names
        df.columns = df.columns.str.strip()
        # Print columns to debug
        print(f"Columns in {file}: {df.columns.tolist()}")
        # Find the correct 'Strike' column (case-insensitive)
        strike_column = next((col for col in df.columns if col.lower() == "strike"), None)
        if strike_column is None:
            print(f"⚠ Error: 'Strike' column not found in {file}. Skipping...")
            continue
        strike_index = df.columns.get_loc(strike_column)
        # Rename columns as per put and call as both are same name on each side of strike price
        new_col_names = {}
        for i, col in enumerate(df.columns):
            if i < strike_index:
                new_col_names[col] = f"Call_{col}"
            elif i > strike_index:
                new_col_names[col] = f"Put_{col}"
        # Column renaming
        df.rename(columns=new_col_names, inplace=True)
        # Saving the files that are cleaned
        output_path = os.path.join(output_folder, file)
        df.to_csv(output_path, index=False)
        print(f"✅ Processed: {file} → {output_path}")
print("All CSV files cleaned and saved in 'cleaned_csvs' folder.")
