from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.chrome.options import Options

from multiprocessing import Process, Queue, Lock

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import yfinance as yf
from time import sleep
from random import shuffle
import sys
import numpy as np

class calendar(Process):
    def __init__(self, date_queue, lock, production, process_num):
        Process.__init__(self)
        self.date_queue = date_queue
        self.lock = lock
        self.production = True
        self.testing = False
        self.process_num = process_num


    def run(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument('log-level=3')
        self.driver = webdriver.Chrome(options=chrome_options)
        self.delay = 30

        self.conn = sqlite3.connect('earnings.db', timeout=120)
        self.cur = self.conn.cursor()
        while self.date_queue.qsize()>0:
            self.start_date, self.end_date = self.date_queue.get(timeout=3)
            total_eps_df = self.get_estimize_data('EPS')
            total_revenue_df = self.get_estimize_data('Revenue')

            df = self.get_combined_df(total_eps_df, total_revenue_df)
            print(df)
            # store in the database
            df.to_sql('estimize_data', self.conn, if_exists='append', index=False)


    def get_estimize_data(self, announcement_type):

        # request the estimize website for data
        if announcement_type == 'EPS':
            url = 'https://www.estimize.com/calendar?tab=equity&startDate=%s&endDate=%s' % (self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))
        elif announcement_type == 'Revenue':
            url = 'https://www.estimize.com/calendar?tab=equity&metric=revenue&startDate=%s&endDate=%s' % (self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))
        self.driver.get(url)
        output_df = []
        previous_first_ticker = ''
        while True:
            # check if there are no companies reporting earnings
            WebDriverWait(self.driver, self.delay).until(EC.presence_of_element_located((By.CLASS_NAME , 'dAViVi')))
            companies_reporting_div = self.driver.find_element_by_class_name('dAViVi')
            if '0 Events' == companies_reporting_div.text.split('\n')[1]:
                print('no events found')
                return

            first_ticker = self.get_first_ticker()
            while first_ticker == previous_first_ticker:
                first_ticker = self.get_first_ticker()
                sleep(.1)
            previous_first_ticker = first_ticker

            # method to extra the ticker symbols from the webpage
            tickers = self.get_tickers()

            df = pd.read_html(self.driver.page_source)[0]
            df['Symbol'] = tickers
            df = df.iloc[:, [2,3,5,6,7,8,9,10,12]]
            df.columns = ['Date Reported', 'Num of Estimates', 'Delta', 'Surprise', 'Historical Beat Rate', 'Wall St', 'Estimize', 'Actual', 'Symbol']

            output_df.append(df)
            df_length = len(pd.concat(output_df))/250
            print('Process Number:', self.process_num, 'Progress:', round(df_length/80, 2))
            if len(pd.concat(output_df))>260 and self.testing:
                return output_df

            try:
                WebDriverWait(self.driver, self.delay).until(EC.presence_of_element_located((By.CLASS_NAME , 'eicEMB')))
                elements = self.driver.find_elements_by_class_name('eicEMB')
                for element in elements:
                    if element.text == 'Next':
                        self.driver.execute_script("arguments[0].click();", element)
            except Exception as e:
                print("Couldn't find next link")
                print(e)
                break

        return output_df

    def get_combined_df(self, total_eps_df, total_revenue_df):
        eps_df = pd.concat(total_eps_df)
        revenue_df = pd.concat(total_eps_df)

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

        df = df.replace('–', np.nan)
        df = df.replace([np.inf, -np.inf], np.nan)

        return df


    def get_first_ticker(self):
        soup = BeautifulSoup(self.driver.page_source, features='lxml')
        ticker_links = soup.findAll('a', attrs={'class': 'lfkTWp'})
        first_ticker = ticker_links[0].contents[0]
        return first_ticker


    def get_tickers(self):
        # extract ticker symbopls from the html source
        soup = BeautifulSoup(self.driver.page_source, features='lxml')
        ticker_links = soup.findAll('a', attrs={'class': 'lfkTWp'})

        # create list of symbols that were extracted
        tickers = []
        for ticker in ticker_links:
            tickers.append(ticker.contents[0])

        return tickers

if __name__ == '__main__':
    if 'production' in sys.argv:
        production = True
        print('Running in production mode.')
    else:
        production = False
        response = input("Run in non-production mode? ")
        if response.lower() != 'y':
            print('Exiting')
            exit()
        else:
            try:
                conn = sqlite3.connect('earnings.db', timeout=120)
                cur = conn.cursor()
                query = 'drop table estimize_data'
                cur.execute(query)
            except:
                pass

    lock = Lock()

    # create queue of dates
    date_queue = Queue()
    date_list = []

    if production == True:
        date_list.append((datetime.now(), datetime.now()))
    else:
        date_list.append((datetime.strptime('2013-01-01', '%Y-%m-%d'), datetime.strptime('2014-12-01', '%Y-%m-%d')))
        date_list.append((datetime.strptime('2014-01-01', '%Y-%m-%d'), datetime.strptime('2014-12-31', '%Y-%m-%d')))
        date_list.append((datetime.strptime('2015-01-01', '%Y-%m-%d'), datetime.strptime('2015-12-31', '%Y-%m-%d')))
        date_list.append((datetime.strptime('2016-01-01', '%Y-%m-%d'), datetime.strptime('2016-12-31', '%Y-%m-%d')))
        date_list.append((datetime.strptime('2017-01-01', '%Y-%m-%d'), datetime.strptime('2017-12-31', '%Y-%m-%d')))
        date_list.append((datetime.strptime('2018-01-01', '%Y-%m-%d'), datetime.strptime('2018-12-31', '%Y-%m-%d')))
        date_list.append((datetime.strptime('2019-01-01', '%Y-%m-%d'), datetime.strptime('2019-12-31', '%Y-%m-%d')))
        date_list.append((datetime.strptime('2020-01-01', '%Y-%m-%d'), datetime.now()))

    #shuffle(date_list)
    for date in date_list:
        date_queue.put(date)

    if production == True:
        num_processes = 1
    else:
        num_processes = 7
    # start the program
    processes = []
    for i in range(num_processes):
        p = calendar(date_queue, lock, production, i+1)
        p.start()
        sleep(30)

    while date_queue.qsize()>0 or processes_running:
        sleep(15)
        print(date_queue.qsize())

        if not any(process.is_alive() for process in processes):
            break
        else:
            print('process running')


    sleep(5)
