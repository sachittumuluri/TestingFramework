import yfinance as yf
import os
import pandas as pd

def download_and_clean(ticker, start, end, filename):
    if not os.path.exists(filename):
        # Download data
        data = yf.download(ticker, start=start, end=end)
        
        # IMPORTANT: yfinance v0.2.x creates a MultiIndex header. 
        # We flatten it so "Close" is just a standard column name.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data.to_csv(filename)
        print(f"Downloaded and saved clean data to: {filename}")

# Re-download your regimes with the fix
download_and_clean('SPY', '2007-10-01', '2009-03-01', '../bearish_SPY_data.csv')
download_and_clean('SPY', '2017-01-01', '2017-12-31', '../bullish_SPY_data.csv')
download_and_clean('SPY', '2015-05-01', '2016-03-01', '../sideways_SPY_data.csv')
download_and_clean('SPY', '2020-02-15', '2020-05-30', '../crash_SPY_data.csv')

print("All regime data files are ready.")