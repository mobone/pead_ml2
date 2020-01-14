import pandas as pd
import sqlite3
from datetime import datetime
import multitasking
import time
import numpy as np
import re
#multitasking.set_engine("process")

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

multitasking.set_max_threads(1)

conn = sqlite3.connect('earnings.db', timeout=120)
total_eps_df = pd.read_sql('select * from estimize_eps', conn)
total_revenue_df = pd.read_sql('select * from estimize_revenue', conn)
total_history_df = pd.read_sql('select * from price_history', conn)

#[['EPS Delta', 'EPS Surprise', 'EPS Estimize']]
total_eps_df = total_eps_df.replace('–', np.nan)

#total_revenue_df = total_revenue_df.applymap(lambda x: re.sub(r'^-$', str(np.NaN), x))


spy_history_df = pd.read_sql('select * from price_history where Symbol == "%s"' % 'SPY', conn)


#@multitasking.task
def aggregator(symbol):
    def get_combined_df(eps_df, revenue_df):
        del eps_df['Historical Beat Rate']
        del revenue_df['Historical Beat Rate']

        date_reported_df = eps_df['Date Reported'].str.split(' ', n = 1, expand = True)
        date_reported_df = date_reported_df.rename(columns={0:"Date Reported", 1:"Time Reported"})
        date_reported_df['Date Reported'] = pd.to_datetime(date_reported_df['Date Reported'])
        eps_df['Date Reported'] = date_reported_df['Date Reported']
        eps_df['Time Reported'] = date_reported_df['Time Reported']

        date_reported_df = revenue_df['Date Reported'].str.split(' ', n = 1, expand = True)
        date_reported_df = date_reported_df.rename(columns={0:"Date Reported", 1:"Time Reported"})
        date_reported_df['Date Reported'] = pd.to_datetime(date_reported_df['Date Reported'])
        revenue_df['Date Reported'] = date_reported_df['Date Reported']
        revenue_df['Time Reported'] = date_reported_df['Time Reported']

        eps_df = eps_df.sort_values(by='Date Reported')
        revenue_df = revenue_df.sort_values(by='Date Reported')

        eps_df = eps_df.set_index(['Date Reported', 'Time Reported', 'Symbol'], append=True, drop = True)
        revenue_df = revenue_df.set_index(['Date Reported', 'Time Reported', 'Symbol'], append=True, drop = True)

        eps_df.columns = 'EPS ' + eps_df.columns
        revenue_df.columns = 'Revenue ' + revenue_df.columns

        df = eps_df.join(revenue_df)

        return df


    def get_price_changes(df, history_df, spy_history_df):
        df['5 Day Change'] = None
        df['10 Day Change'] = None
        df['5 Day Change Abnormal'] = None
        df['10 Day Change Abnormal'] = None

        history_df['Date'] = pd.to_datetime(history_df['Date'])
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        spy_history_df['Date'] = pd.to_datetime(spy_history_df['Date'])


        spy_history = spy_history_df
        spy_history['Open to Close 5 Days'] = (spy_history['Close'] / spy_history['Open'].shift(4) - 1)
        spy_history['Open to Close 10 Days'] = (spy_history['Close'] / spy_history['Open'].shift(9) - 1)


        for index, row in df.iterrows():
            index_num, date_reported, time_reported, symbol = index

            this_history = history_df[history_df['Symbol'] == symbol]
            this_history['Open to Close 5 Days'] = (this_history['Close'] / this_history['Open'].shift(4) - 1)
            this_history['Open to Close 10 Days'] = (this_history['Close'] / this_history['Open'].shift(9) - 1)


            try:
                if 'BMO' in time_reported:
                    percent_change = this_history[this_history['Date'] == date_reported]['Open to Close 5 Days'].values[0]
                    spy_percent_change = spy_history[spy_history['Date'] == date_reported]['Open to Close 5 Days'].values[0]
                    df.loc[index_num, ['5 Day Change']] = percent_change
                    df.loc[index_num, ['5 Day Change Abnormal']] = percent_change - spy_percent_change

                    percent_change = this_history[this_history['Date'] == date_reported]['Open to Close 10 Days'].values[0]
                    spy_percent_change = spy_history[spy_history['Date'] == date_reported]['Open to Close 10 Days'].values[0]
                    df.loc[index_num, ['10 Day Change']] = percent_change
                    df.loc[index_num, ['10 Day Change Abnormal']] = percent_change - spy_percent_change
                else:
                    percent_change = this_history[this_history['Date'] > date_reported]['Open to Close 5 Days'].head(1).values[0]
                    spy_percent_change = spy_history[spy_history['Date'] > date_reported]['Open to Close 5 Days'].head(1).values[0]
                    df.loc[index_num, ['5 Day Change']] = percent_change
                    df.loc[index_num, ['5 Day Change Abnormal']] = percent_change - spy_percent_change

                    percent_change = this_history[this_history['Date'] > date_reported]['Open to Close 10 Days'].head(1).values[0]
                    spy_percent_change = spy_history[spy_history['Date'] > date_reported]['Open to Close 10 Days'].head(1).values[0]
                    df.loc[index_num, ['10 Day Change']] = percent_change
                    df.loc[index_num, ['10 Day Change Abnormal']] = percent_change - spy_percent_change
            except Exception as e:
                pass


    def get_historical_beat():
        df['Historical EPS Beat Ratio'] = None
        df['Historical EPS Beat Percent'] = None
        for index, row in df.iterrows():
            index_num, date_reported, time_reported, symbol = index

            this_df = df[df.index.get_level_values('Symbol')==symbol]
            beat_rate = this_df[this_df.index.get_level_values('Date Reported') < date_reported].tail(8)

            if len(beat_rate)<4:
                beat_rate_percent = None
                beat_rate_ratio = None
            else:

                beat_rate_ratio = len(beat_rate[beat_rate['EPS Surprise'] > 0]) / float(len(beat_rate))
                beat_rate_percent = beat_rate['EPS Surprise'] / beat_rate['EPS Actual']
                beat_rate_percent = beat_rate_percent.replace([np.inf, -np.inf], np.nan)
                beat_rate_percent = beat_rate_percent.mean()


            # TODO: Do the same for revenue
            df.loc[index_num, ['Historical EPS Beat Ratio']] = beat_rate_ratio
            df.loc[index_num, ['Historical EPS Beat Percent']] = beat_rate_percent


    def get_average_change():
        df['Average Change 5 Days'] = None
        df['Average Abnormal Change 5 Days'] = None
        df['Average Change 10 Days'] = None
        df['Average Abnormal Change 10 Days'] = None
        for index, row in df.iterrows():
            index_num, date_reported, time_reported, symbol = index

            this_df = df[df.index.get_level_values('Symbol')==symbol]
            returns_df = this_df[this_df.index.get_level_values('Date Reported') < date_reported].tail(8)

            if len(returns_df)>=4:
                df.loc[index_num, ['Average Change 5 Days']] = returns_df['5 Day Change'].mean()
                df.loc[index_num, ['Average Change 10 Days']] = returns_df['10 Day Change'].mean()
                df.loc[index_num, ['Average Abnormal Change 5 Days']] = returns_df['5 Day Change Abnormal'].mean()
                df.loc[index_num, ['Average Abnormal Change 10 Days']] = returns_df['10 Day Change Abnormal'].mean()


    def get_YoY_growth():
        df['YoY Growth'] = None
        for index, row in df.iterrows():
            index_num, date_reported, time_reported, symbol = index
            time_reported = time_reported.replace("'",'')
            quarter_numer, year = time_reported.split(' ')

            this_df = df['EPS Actual']
            try:
                this_quarter = this_df[this_df.index.get_level_values('Time Reported') == quarter_numer + " '" + year].values[0]
                last_quarter = this_df[this_df.index.get_level_values('Time Reported') == quarter_numer + " '" + str(int(year)-1)].values[0]
                df.loc[index_num, ['YoY Growth']] = (this_quarter - last_quarter) / last_quarter
            except Exception as e:
                pass


    def get_market_cap():
        pass
    start_time = time.time()
    conn = sqlite3.connect('earnings.db', timeout=120)
    cur = conn.cursor()
    #symbol = 'BBY'

    eps_df = total_eps_df[total_eps_df['Symbol'] == symbol]
    revenue_df = total_revenue_df[total_revenue_df['Symbol'] == symbol]
    history_df = total_history_df[total_history_df['Symbol'] == symbol]

    df = get_combined_df(eps_df, revenue_df)

    get_price_changes(df, history_df, spy_history_df)

    get_historical_beat()
    get_average_change()
    get_YoY_growth()
    get_market_cap()

    df.to_sql('aggregated_data_verified', conn, if_exists='append')

    print((time.time()-start_time)*2921)

if __name__ == '__main__':


    conn = sqlite3.connect('earnings.db', timeout=120)
    symbols = pd.read_sql('select DISTINCT symbol from estimize_eps;', conn)
    symbols = symbols.values.tolist()

    """
    for symbol in symbols:
        aggregator(symbol[0])
    """
    aggregator('NEOG')



    #aggregator('BBY')
