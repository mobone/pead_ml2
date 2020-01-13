import pandas as pd
import sqlite3
from datetime import datetime

pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class aggregator():
    def __init__(self):
        conn = sqlite3.connect('earnings.db', timeout=120)
        cur = conn.cursor()
        symbol = 'BBY'
        self.eps_df = pd.read_sql('select * from estimize_eps where Symbol == "%s"' % symbol, conn)
        self.revenue_df = pd.read_sql('select * from estimize_revenue where Symbol == "%s"' % symbol, conn)
        self.history_df = pd.read_sql('select * from price_history_2 where Symbol == "%s"' % symbol, conn)

        self.spy_history_df = pd.read_sql('select * from price_history_2 where Symbol == "%s"' % 'SPY', conn)

        self.get_combined_df()
        self.get_price_changes()

        self.get_historical_beat()
        self.get_average_change()
        self.get_YoY_growth()
        self.get_market_cap()

        self.df[self.df.index.get_level_values('Symbol')==symbol].to_csv('earnings_df.csv')
        #this_history[this_history['Symbol']==symbol].to_csv('history_df.csv')


    def get_combined_df(self):
        del self.eps_df['Historical Beat Rate']
        del self.revenue_df['Historical Beat Rate']

        date_reported_df = self.eps_df['Date Reported'].str.split(' ', n = 1, expand = True)
        date_reported_df.columns = ['Date Reported', 'Time Reported']
        date_reported_df['Date Reported'] = pd.to_datetime(date_reported_df['Date Reported'])
        self.eps_df['Date Reported'] = date_reported_df['Date Reported']
        self.eps_df['Time Reported'] = date_reported_df['Time Reported']

        date_reported_df = self.revenue_df['Date Reported'].str.split(' ', n = 1, expand = True)
        date_reported_df.columns = ['Date Reported', 'Time Reported']
        date_reported_df['Date Reported'] = pd.to_datetime(date_reported_df['Date Reported'])
        self.revenue_df['Date Reported'] = date_reported_df['Date Reported']
        self.revenue_df['Time Reported'] = date_reported_df['Time Reported']

        self.eps_df = self.eps_df.set_index(['Date Reported', 'Time Reported', 'Symbol'], append=True, drop = True)
        self.revenue_df = self.revenue_df.set_index(['Date Reported', 'Time Reported', 'Symbol'], append=True, drop = True)

        self.eps_df.columns = 'EPS ' + self.eps_df.columns
        self.revenue_df.columns = 'Revenue ' + self.revenue_df.columns

        self.df = self.eps_df.join(self.revenue_df)


    def get_price_changes(self):
        self.df['5 Day Change'] = None
        self.df['10 Day Change'] = None
        self.df['5 Day Change Abnormal'] = None
        self.df['10 Day Change Abnormal'] = None

        self.history_df['Date'] = pd.to_datetime(self.history_df['Date'])
        self.history_df['Date'] = pd.to_datetime(self.history_df['Date'])
        self.spy_history_df['Date'] = pd.to_datetime(self.spy_history_df['Date'])


        spy_history = self.spy_history_df
        spy_history['Open to Close 5 Days'] = (spy_history['Close'] / spy_history['Open'].shift(4) - 1)
        spy_history['Open to Close 10 Days'] = (spy_history['Close'] / spy_history['Open'].shift(9) - 1)


        for index, row in self.df.iterrows():
            index_num, date_reported, time_reported, symbol = index

            this_history = self.history_df[self.history_df['Symbol'] == symbol]
            this_history['Open to Close 5 Days'] = (this_history['Close'] / this_history['Open'].shift(4) - 1)
            this_history['Open to Close 10 Days'] = (this_history['Close'] / this_history['Open'].shift(9) - 1)



            if 'BMO' in time_reported:
                percent_change = this_history[this_history['Date'] == date_reported]['Open to Close 5 Days'].values[0]
                spy_percent_change = spy_history[spy_history['Date'] == date_reported]['Open to Close 5 Days'].values[0]
                self.df.iloc[index_num, self.df.columns.get_loc('5 Day Change')] = percent_change
                self.df.iloc[index_num, self.df.columns.get_loc('5 Day Change Abnormal')] = percent_change - spy_percent_change

                percent_change = this_history[this_history['Date'] == date_reported]['Open to Close 10 Days'].values[0]
                spy_percent_change = spy_history[spy_history['Date'] == date_reported]['Open to Close 10 Days'].values[0]
                self.df.iloc[index_num, self.df.columns.get_loc('10 Day Change')] = percent_change
                self.df.iloc[index_num, self.df.columns.get_loc('10 Day Change Abnormal')] = percent_change - spy_percent_change
            else:
                percent_change = this_history[this_history['Date'] > date_reported]['Open to Close 5 Days'].head(1).values[0]
                spy_percent_change = spy_history[spy_history['Date'] > date_reported]['Open to Close 5 Days'].head(1).values[0]
                self.df.iloc[index_num, self.df.columns.get_loc('5 Day Change')] = percent_change
                self.df.iloc[index_num, self.df.columns.get_loc('5 Day Change Abnormal')] = percent_change - spy_percent_change

                percent_change = this_history[this_history['Date'] > date_reported]['Open to Close 10 Days'].head(1).values[0]
                spy_percent_change = spy_history[spy_history['Date'] > date_reported]['Open to Close 10 Days'].head(1).values[0]
                self.df.iloc[index_num, self.df.columns.get_loc('10 Day Change')] = percent_change
                self.df.iloc[index_num, self.df.columns.get_loc('10 Day Change Abnormal')] = percent_change - spy_percent_change

            # TODO: Buy before announcement price changes
        print(self.df)


    def get_historical_beat(self):
        self.df['Historical EPS Beat Ratio'] = None
        self.df['Historical EPS Beat Percent'] = None
        for index, row in self.df.iterrows():
            index_num, date_reported, time_reported, symbol = index

            this_df = self.df[self.df.index.get_level_values('Symbol')==symbol]
            beat_rate = this_df[this_df.index.get_level_values('Date Reported') < date_reported].head(8)

            if len(beat_rate)<4:
                beat_rate_percent = None
                beat_rate_ratio = None
            else:
                beat_rate_ratio = len(beat_rate[beat_rate['EPS Surprise'] > 0]) / float(len(beat_rate))
                beat_rate_percent = beat_rate['EPS Surprise'] / beat_rate['EPS Actual']
                beat_rate_percent = beat_rate_percent.mean()


            self.df.iloc[index_num, self.df.columns.get_loc('Historical EPS Beat Ratio')] = beat_rate_ratio
            self.df.iloc[index_num, self.df.columns.get_loc('Historical EPS Beat Percent')] = beat_rate_percent


    def get_average_change(self):
        self.df['Average Change 5 Days'] = None
        self.df['Average Abnormal Change 5 Days'] = None
        self.df['Average Change 10 Days'] = None
        self.df['Average Abnormal Change 10 Days'] = None
        for index, row in self.df.iterrows():
            index_num, date_reported, time_reported, symbol = index

            this_df = self.df[self.df.index.get_level_values('Symbol')==symbol]
            returns_df = this_df[this_df.index.get_level_values('Date Reported') < date_reported].head(8)

            if len(returns_df)>=4:
                self.df.iloc[index_num, self.df.columns.get_loc('Average Change 5 Days')] = returns_df['5 Day Change'].mean()
                self.df.iloc[index_num, self.df.columns.get_loc('Average Change 10 Days')] = returns_df['10 Day Change'].mean()
                self.df.iloc[index_num, self.df.columns.get_loc('Average Abnormal Change 5 Days')] = returns_df['5 Day Change Abnormal'].mean()
                self.df.iloc[index_num, self.df.columns.get_loc('Average Abnormal Change 10 Days')] = returns_df['10 Day Change Abnormal'].mean()


    def get_YoY_growth(self):
        self.df['YoY Growth'] = None
        for index, row in self.df.iterrows():
            index_num, date_reported, time_reported, symbol = index
            time_reported = time_reported.replace("'",'')
            quarter_numer, year = time_reported.split(' ')


            this_df = self.df['EPS Actual']
            #BMOFQ3 '20 == BMOFQ3 '19
            print(this_df)
            print(quarter_numer + " '" + year)
            try:
                this_quarter = this_df[this_df.index.get_level_values('Time Reported') == quarter_numer + " '" + year].values[0]
                last_quarter = this_df[this_df.index.get_level_values('Time Reported') == quarter_numer + " '" + str(int(year)-1)].values[0]
            except Exception as e:
                print(e)
                input()
                pass




    def get_market_cap(self):
        pass






aggregator()
