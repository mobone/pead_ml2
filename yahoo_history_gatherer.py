import pandas as pd
import sqlite3
import yfinance as yf

class yahoo_gather():
    def __init__(self):
        self.conn = sqlite3.connect('earnings.db', timeout=120)

        self.get_symbols()
        self.get_yahoo_historical()

    def get_symbols(self):
        symbols_df = pd.read_sql('select DISTINCT symbol from estimize_eps', self.conn)
        self.symbols = symbols_df['Symbol'].tolist()
        self.symbols.append('SPY')



    def get_yahoo_historical(self):

        # query yahoo finance for the historical data
        data = yf.download(self.symbols, start="2012-01-01", end="2019-12-31", group_by = 'ticker', auto_adjust = False)

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

yahoo_gather()