import pandas as pd
import sqlite3
from datetime import datetime
import multitasking
import time
import numpy as np
import re
import requests as r
from bs4 import BeautifulSoup
import requests_cache
import warnings
import sys
from random import shuffle
import talib
requests_cache.install_cache('finviz_cache')

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

multitasking.set_max_threads(2)

conn = sqlite3.connect('earnings.db', timeout=120)
total_eps_df = pd.read_sql('select * from estimize_eps', conn)
total_revenue_df = pd.read_sql('select * from estimize_revenue', conn)
total_history_df = pd.read_sql('select * from price_history_adjusted', conn, parse_dates=['Date'])
spy_history_df = total_history_df[total_history_df['Symbol']=='SPY']

total_eps_df = total_eps_df.replace('–', np.nan)
total_revenue_df = total_revenue_df.replace('–', np.nan)

total_eps_df = total_eps_df.drop_duplicates(subset=['Date Reported', 'Symbol'])
total_revenue_df = total_revenue_df.drop_duplicates(subset=['Date Reported', 'Symbol'])


#@multitasking.task
def aggregator(symbol):
    def get_combined_df(eps_df, revenue_df):
        eps_df = eps_df.sort_values(by='Date Reported')
        revenue_df = revenue_df.sort_values(by='Date Reported')

        eps_df = eps_df.set_index(['Date Reported', 'Time Reported', 'Symbol'], drop = True)
        revenue_df = revenue_df.set_index(['Date Reported', 'Time Reported', 'Symbol'], drop = True)

        eps_df.columns = 'EPS ' + eps_df.columns
        revenue_df.columns = 'Revenue ' + revenue_df.columns

        df = eps_df.join(revenue_df)

        return df

    def get_before_announcement_change():
        df['Pre Announcement 5 Day Change'] = None
        df['Pre Announcement 10 Day Change'] = None

        for index, row in df.iterrows():
            date_reported, time_reported, symbol = index

            history_df['Open to Close 5 Days'] = history_df['Open to Close 5 Days'].shift(4)
            history_df['Open to Close 10 Days'] = history_df['Open to Close 10 Days'].shift(9)

            try:
                if 'BMO' in time_reported:
                    percent_change = history_df[history_df['Date'] < date_reported]['Open to Close 5 Days'].tail(1).values[0]
                    spy_percent_change = spy_history_df[spy_history_df['Date'] < date_reported]['Open to Close 5 Days'].tail(1).values[0]
                    df.loc[index, 'Pre Announcement 5 Day Change'] = percent_change - spy_percent_change

                    percent_change = history_df[history_df['Date'] < date_reported]['Open to Close 10 Days'].tail(1).values[0]
                    spy_percent_change = spy_history_df[spy_history_df['Date'] < date_reported]['Open to Close 10 Days'].tail(1).values[0]
                    df.loc[index, 'Pre Announcement 10 Day Change'] = percent_change - spy_percent_change
                else:
                    percent_change = history_df[history_df['Date'] == date_reported]['Open to Close 5 Days'].values[0]
                    spy_percent_change = spy_history_df[spy_history_df['Date'] == date_reported]['Open to Close 5 Days'].values[0]
                    df.loc[index, 'Pre Announcement 5 Day Change'] = percent_change - spy_percent_change

                    percent_change = history_df[history_df['Date'] == date_reported]['Open to Close 10 Days'].values[0]
                    spy_percent_change = spy_history_df[spy_history_df['Date'] == date_reported]['Open to Close 10 Days'].values[0]
                    df.loc[index, 'Pre Announcement 10 Day Change'] = percent_change - spy_percent_change


            except Exception as e:
                print(e)
                pass


    def get_ta():
        try:
            history_df['MOM 5 Days'] = talib.MOM(history_df['Close'], timeperiod=5)
            history_df['MOM 10 Days'] = talib.MOM(history_df['Close'], timeperiod=10)
            history_df['RSI 5 Days'] = talib.RSI(history_df['Close'], timeperiod=5)
            history_df['RSI 10 Days'] = talib.RSI(history_df['Close'], timeperiod=10)
        except:
            return
        for index, row in df.iterrows():
            date_reported, time_reported, symbol = index
            try:
                if 'BMO' in time_reported:
                    mom = history_df[history_df['Date'] < date_reported]['MOM 5 Days'].tail(1).values[0]
                    df.loc[index, 'MOM 5 Days'] = mom
                    mom = history_df[history_df['Date'] < date_reported]['MOM 10 Days'].tail(1).values[0]
                    df.loc[index, 'MOM 10 Days'] = mom
                    rsi = history_df[history_df['Date'] < date_reported]['RSI 5 Days'].tail(1).values[0]
                    df.loc[index, 'RSI 5 Days'] = rsi
                    rsi = history_df[history_df['Date'] < date_reported]['RSI 10 Days'].tail(1).values[0]
                    df.loc[index, 'RSI 10 Days'] = rsi
                else:
                    mom = history_df[history_df['Date'] == date_reported]['MOM 5 Days'].values[0]
                    df.loc[index, 'MOM 5 Days'] = mom
                    mom = history_df[history_df['Date'] == date_reported]['MOM 10 Days'].values[0]
                    df.loc[index, 'MOM 10 Days'] = mom
                    rsi = history_df[history_df['Date'] == date_reported]['RSI 5 Days'].values[0]
                    df.loc[index, 'RSI 5 Days'] = rsi
                    rsi = history_df[history_df['Date'] == date_reported]['RSI 10 Days'].values[0]
                    df.loc[index, 'RSI 10 Days'] = rsi
            except Exception as e:
                print(e)
                pass


    def get_price_changes():
        #df['Start Price'] = None
        #df['End Price'] = None
        df['5 Day Change'] = None
        df['10 Day Change'] = None
        df['5 Day Change Abnormal'] = None
        df['10 Day Change Abnormal'] = None

        #history_df['Date'] = pd.to_datetime(history_df['Date'])
        #spy_history_df['Date'] = pd.to_datetime(spy_history_df['Date'])



        spy_history_df['Open to Close 5 Days'] = (spy_history_df['Close'] / spy_history_df['Open'].shift(4) - 1).shift(-4)
        spy_history_df['Open to Close 10 Days'] = (spy_history_df['Close'] / spy_history_df['Open'].shift(9) - 1).shift(-9)

        history_df['Open to Close 5 Days'] = (history_df['Close'] / history_df['Open'].shift(4) - 1).shift(-4)
        history_df['Open to Close 10 Days'] = (history_df['Close'] / history_df['Open'].shift(9) - 1).shift(-9)
        #history_df.to_csv('history.csv')

        for index, row in df.iterrows():
            date_reported, time_reported, symbol = index



            try:
                if 'BMO' in time_reported:
                    #start_price = history_df[history_df['Date'] == date_reported]['Open'].values[0]
                    #end_price = history_df[history_df['Date'] == date_reported]['Close'].values[0]
                    #df.loc[(date_reported, time_reported, symbol), ['Start Price']] = start_price
                    #df.loc[(date_reported, time_reported, symbol), ['End Price']] = end_price

                    percent_change = history_df[history_df['Date'] == date_reported]['Open to Close 5 Days'].values[0]
                    spy_percent_change = spy_history_df[spy_history_df['Date'] == date_reported]['Open to Close 5 Days'].values[0]
                    df.loc[(date_reported, time_reported, symbol), ['5 Day Change']] = percent_change
                    df.loc[(date_reported, time_reported, symbol), ['5 Day Change Abnormal']] = percent_change - spy_percent_change

                    percent_change = history_df[history_df['Date'] == date_reported]['Open to Close 10 Days'].values[0]
                    spy_percent_change = spy_history_df[spy_history_df['Date'] == date_reported]['Open to Close 10 Days'].values[0]
                    df.loc[(date_reported, time_reported, symbol), ['10 Day Change']] = percent_change
                    df.loc[(date_reported, time_reported, symbol), ['10 Day Change Abnormal']] = percent_change - spy_percent_change
                else:
                    #history_date = history_df[history_df['Date'] > date_reported]['Date'].head(1)
                    # TODO: Verify AMC is actually using the next day prices and not some time way in the future
                    # deal with head(1)
                    #start_price = history_df[history_df['Date'] > date_reported]['Open'].head(1).values[0]
                    #end_price = history_df[history_df['Date'] > date_reported]['Close'].head(1).values[0]
                    #df.loc[(date_reported, time_reported, symbol), ['Start Price']] = start_price
                    #df.loc[(date_reported, time_reported, symbol), ['End Price']] = end_price


                    history_date = history_df[history_df['Date'] > date_reported]['Date'].head(1).values[0]
                    days_between = (pd.to_datetime(date_reported.split(' ')[0], format='%Y-%m-%d') - history_date).days
                    if abs(days_between) > 3:
                        return

                    percent_change = history_df[history_df['Date'] > date_reported]['Open to Close 5 Days'].head(1).values[0]
                    spy_percent_change = spy_history_df[spy_history_df['Date'] > date_reported]['Open to Close 5 Days'].head(1).values[0]
                    df.loc[(date_reported, time_reported, symbol), ['5 Day Change']] = percent_change
                    df.loc[(date_reported, time_reported, symbol), ['5 Day Change Abnormal']] = percent_change - spy_percent_change

                    percent_change = history_df[history_df['Date'] > date_reported]['Open to Close 10 Days'].head(1).values[0]
                    spy_percent_change = spy_history_df[spy_history_df['Date'] > date_reported]['Open to Close 10 Days'].head(1).values[0]
                    df.loc[(date_reported, time_reported, symbol), ['10 Day Change']] = percent_change
                    df.loc[(date_reported, time_reported, symbol), ['10 Day Change Abnormal']] = percent_change - spy_percent_change
            except Exception as e:
                pass


    def get_before_and_after_change():
        df['Before and After'] = None
        history_df['Before and After'] = (history_df['Open'] / history_df['Close'].shift(1) - 1).shift(-1)

        for index, row in df.iterrows():
            date_reported, time_reported, symbol = index

            try:
                if "BMO" in time_reported:
                    percent_change = history_df[history_df['Date'] == date_reported]['Before and After'].values[0]
                else:
                    history_date = history_df[history_df['Date'] > date_reported]['Date'].head(1).values[0]
                    days_between = (pd.to_datetime(date_reported.split(' ')[0], format='%Y-%m-%d') - history_date).days
                    if abs(days_between) > 3:
                        return
                    percent_change = history_df[history_df['Date'] > date_reported]['Before and After'].values[0]


                df.loc[(date_reported, time_reported, symbol), ['Before and After']] =percent_change
            except:
                pass


    def get_historical_beat():
        df['Historical EPS Beat Ratio'] = None
        df['Historical EPS Beat Percent'] = None
        df['Historical Revenue Beat Ratio'] = None
        df['Historical Revenue Beat Percent'] = None
        for index, row in df.iterrows():

            date_reported, time_reported, symbol = index


            beat_rate = df[df.index.get_level_values('Date Reported') <= date_reported].tail(8)

            if len(beat_rate)>=4:
                beat_rate_ratio = len(beat_rate[beat_rate['EPS Surprise'] > 0]) / float(len(beat_rate))
                beat_rate_percent = beat_rate['EPS Surprise'] / beat_rate['EPS Actual']
                beat_rate_percent = beat_rate_percent.replace([np.inf, -np.inf], np.nan)
                beat_rate_percent = beat_rate_percent.mean()

                df.loc[(date_reported, time_reported, symbol), ['Historical EPS Beat Ratio']] = beat_rate_ratio
                df.loc[(date_reported, time_reported, symbol), ['Historical EPS Beat Percent']] = beat_rate_percent

                beat_rate_ratio = len(beat_rate[beat_rate['Revenue Surprise'] > 0]) / float(len(beat_rate))
                beat_rate_percent = beat_rate['Revenue Surprise'] / beat_rate['Revenue Actual']
                beat_rate_percent = beat_rate_percent.replace([np.inf, -np.inf], np.nan)
                beat_rate_percent = beat_rate_percent.mean()

                df.loc[(date_reported, time_reported, symbol), ['Historical Revenue Beat Ratio']] = beat_rate_ratio
                df.loc[(date_reported, time_reported, symbol), ['Historical Revenue Beat Percent']] = beat_rate_percent


    def get_average_change():
        df['Average Change 5 Days'] = None
        df['Average Abnormal Change 5 Days'] = None
        df['Average Change 10 Days'] = None
        df['Average Abnormal Change 10 Days'] = None
        for index, row in df.iterrows():
            date_reported, time_reported, symbol = index

            returns_df = df[df.index.get_level_values('Date Reported') < date_reported].tail(8)

            if len(returns_df)>=4:
                df.loc[(date_reported, time_reported, symbol), ['Average Change 5 Days']] = returns_df['5 Day Change'].mean()
                df.loc[(date_reported, time_reported, symbol), ['Average Change 10 Days']] = returns_df['10 Day Change'].mean()
                df.loc[(date_reported, time_reported, symbol), ['Average Abnormal Change 5 Days']] = returns_df['5 Day Change Abnormal'].mean()
                df.loc[(date_reported, time_reported, symbol), ['Average Abnormal Change 10 Days']] = returns_df['10 Day Change Abnormal'].mean()

    def get_sue():
        df['EPS SUE'] = None
        df['Revenue SUE'] = None
        df['EPS SUE Ratio'] = None
        df['Revenue SUE Ratio'] = None
        for index, row in df.iterrows():
            eps_sue = (row['EPS Actual'] - row['EPS Estimize'])/np.std([row['EPS Estimize'], row['EPS Wall St']])
            eps_sue_ratio = row['EPS Surprise Percent'] / eps_sue

            revenue_sue = (row['Revenue Actual'] - row['Revenue Estimize'])/np.std([row['Revenue Estimize'], row['Revenue Wall St']])
            revenue_sue_ratio = row['Revenue Surprise Percent'] / eps_sue

            df.loc[index, ['EPS SUE']] = eps_sue
            df.loc[index, ['Revenue SUE']] = revenue_sue
            df.loc[index, ['EPS SUE Ratio']] = eps_sue_ratio
            df.loc[index, ['Revenue SUE Ratio']] = revenue_sue_ratio




    def get_YoY_growth():
        df['YoY Growth'] = None
        for index, row in df.iterrows():

            date_reported, time_reported, symbol = index
            quarter_numer, year = time_reported.split(' ')
            year = year.replace("'",'')

            this_df = df['EPS Actual']
            try:
                this_quarter = this_df[this_df.index.get_level_values('Time Reported') == quarter_numer + " '" + year].values[0]
                last_quarter = this_df[this_df.index.get_level_values('Time Reported') == quarter_numer + " '" + str(int(year)-1)].values[0]
                df.loc[(date_reported, time_reported, symbol), ['YoY Growth']] = (this_quarter - last_quarter) / last_quarter
            except Exception as e:
                pass
    def get_surprise_percent():
        df['EPS Surprise Percent'] = df['EPS Surprise']/df['EPS Actual']
        df['Revenue Surprise Percent'] = df['Revenue Surprise']/df['Revenue Actual']

    def get_market_cap():
        finviz_page = r.get('https://finviz.com/quote.ashx?t=%s' % symbol)

        soup = BeautifulSoup(finviz_page.text, features='lxml')
        try:
            table_row = soup.findAll('tr', attrs={'class': "table-dark-row"})[1]
        except:
            return
        market_cap = table_row.text.replace('Market Cap','').split('\n')[1]
        if 'K' in market_cap:
            market_cap = float(market_cap[:-1])*1000
        elif 'M' in market_cap:
            market_cap = float(market_cap[:-1])*1000000
        elif 'B' in market_cap:
            market_cap = float(market_cap[:-1])*1000000000

        try:
            market_cap = int(market_cap)
        except:
            return
        if market_cap > 10000000000:
            market_cap_text = 'Large'
        elif market_cap > 2000000000:
            market_cap_text = 'Medium'
        elif market_cap > 300000000:
            market_cap_text = 'Small'
        elif market_cap > 50000000:
            market_cap_text = 'Micro'
        else:
            market_cap_text = 'Nano'

        df['Market Cap Text'] = market_cap_text


    start_time = time.time()
    conn = sqlite3.connect('earnings.db', timeout=120)
    cur = conn.cursor()

    symbol = symbol[0]
    print(symbol)

    eps_df = total_eps_df[total_eps_df['Symbol'] == symbol]
    revenue_df = total_revenue_df[total_revenue_df['Symbol'] == symbol]
    history_df = total_history_df[total_history_df['Symbol'] == symbol]


    df = get_combined_df(eps_df, revenue_df)

    get_historical_beat()
    get_market_cap()
    get_YoY_growth()
    get_price_changes()
    get_before_and_after_change()
    get_surprise_percent()
    #get_sue()
    get_average_change()
    get_before_announcement_change()
    get_ta()


    """
    if production == True:
        #df = df.tail(1)
        df = df.iloc[df.index.get_level_values('Date Reported') == datetime.now().strftime('%Y-%m-%d') + ' 00:00:00']
    """


    df.to_sql('aggregated_data_adjusted_ta', conn, if_exists='append')
    print(df.tail())

    #print((time.time()-start_time)*2921)

def drop_table(table_name):
    conn = sqlite3.connect('earnings.db', timeout=120)
    cur = conn.cursor()
    try:
        query = 'drop table %s' % table_name
        cur.execute(query)
        print('Dropped table', table_name)
    except Exception as e:
        print('Failed to drop table', table_name)
        pass

def get_symbols(production):
    conn = sqlite3.connect('earnings.db', timeout=120)
    cur = conn.cursor()
    if production == True:
        date_reported = datetime.now().strftime('%Y-%m-%d')
        symbols = pd.read_sql('select DISTINCT symbol from estimize_eps where "Date Reported">="%s";' % date_reported, conn)
    else:
        symbols = pd.read_sql('select DISTINCT symbol from estimize_eps;', conn)
    symbols = symbols.values.tolist()
    return symbols


if __name__ == '__main__':
    conn = sqlite3.connect('earnings.db', timeout=120)
    cur = conn.cursor()

    announcement_time = None
    if 'production' in sys.argv:
        production = True
        announcement_time = sys.argv[2]
    else:
        production = False
        response = input("Run in non-production mode? ")
        if response.lower() != 'y':
            print('Exiting')
            exit()
        else:
            response = input("Drop Table? ")
            if response == 'y':
                drop_table('aggregated_data_adjusted_ta')

    symbols = get_symbols(production)
    #aggregator(['BYND'])

    from multiprocessing import Pool
    p = Pool(6)
    p.map(aggregator, symbols)
