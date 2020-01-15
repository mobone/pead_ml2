import pandas as pd
import sqlite3
import yfinance as yf
from datetime import datetime

class yahoo_gather():
    def __init__(self):
        self.conn = sqlite3.connect('earnings.db', timeout=120)

        self.get_symbols()
        self.get_yahoo_historical()

    def get_symbols(self):
        symbols_df = pd.read_sql('select DISTINCT symbol from estimize_eps', self.conn)
        self.symbols = symbols_df['Symbol'].tolist()
        self.symbols.append('SPY')
        self.symbols.append('DJI')



    def get_yahoo_historical(self):

        # query yahoo finance for the historical data
        data = yf.download(self.symbols, start="2013-01-01", end=datetime.now().strftime('%Y-%m-%d'), group_by = 'ticker', auto_adjust = False)

        # TODO: Make this not iterative
        for symbol in data.columns:
            if symbol[1] == 'Open':
                symbol = symbol[0]
                df = data[symbol]
                df['Symbol'] = symbol
                df = df.dropna()
                print(df)

                # store in the database
                df.to_sql('price_history', self.conn, if_exists='append')

# TODO: add production mode to gather history daily
yahoo_gather()
