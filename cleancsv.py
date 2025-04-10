import os
import pandas as pd

# Folder containing the original CSV files
input_folder = r"E:\Stocks\Options csv"  # Change to your folder path
output_folder = os.path.join(input_folder, "cleaned_csvs")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each CSV file in the folder
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(input_folder, file)

        # Read CSV again, now using the SECOND row as the header
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

        # Find the index of the 'Strike' column
        strike_index = df.columns.get_loc(strike_column)

        # Rename columns: Calls (left of Strike), Puts (right of Strike)
        new_col_names = {}
        for i, col in enumerate(df.columns):
            if i < strike_index:
                new_col_names[col] = f"Call_{col}"
            elif i > strike_index:
                new_col_names[col] = f"Put_{col}"

        # Rename the columns
        df.rename(columns=new_col_names, inplace=True)

        # Save cleaned file
        output_path = os.path.join(output_folder, file)
        df.to_csv(output_path, index=False)
        print(f"✅ Processed: {file} → {output_path}")

print("🎉 All CSV files cleaned and saved in 'cleaned_csvs' folder.")
