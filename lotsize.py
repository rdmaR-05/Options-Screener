import requests
from bs4 import BeautifulSoup

# Replace with the actual URL
url = 'https://dhan.co/nse-fno-lot-size/'  # Update this with the actual website URL

# Send a GET request to the website
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all stock symbols and lot sizes
symbols = soup.find_all('td', class_='font-CircularRegular relative text-[#666666] lg:text-sm text-xs')
lot_sizes = soup.find_all('td', class_='font-CircularRegular text-[#666666] lg:text-sm text-xs')

# Create the dictionary
lot_size_dict = {}

# Assuming the symbol and lot size appear in sequence
for symbol, lot_size in zip(symbols, lot_sizes):
    symbol_text = symbol.find('p').text.strip() if symbol.find('p') else None
    lot_size_text = lot_size.text.strip()

    if symbol_text and lot_size_text.isdigit():
        lot_size_dict[symbol_text] = int(lot_size_text)

# Display the dictionary
# print(lot_size_dict)
