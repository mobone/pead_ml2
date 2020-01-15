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
import time
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
        self.delay = 10

        self.conn = sqlite3.connect('earnings.db', timeout=120)
        self.cur = self.conn.cursor()
        while self.date_queue.qsize()>0:
            self.start_date, self.end_date = self.date_queue.get(timeout=3)

            #self.get_estimize_data('EPS')
            #self.get_estimize_data('Revenue')

            df = self.get_combined_df()
            print('Output df', df)
            # store in the database
            df.to_sql('estimize_data', self.conn, if_exists='append', index=True)


    def get_estimize_data(self, announcement_type):
        print('Process Num', self.process_num, 'starting to gather', announcement_type, self.start_date.strftime('%Y-%m-%d'))
        max_page_num = 1000
        page_num = 1
        previous_first_ticker = ''
        while page_num<=max_page_num:
            self.start_time = time.time()
            # request the estimize website for data
            if announcement_type == 'EPS':
                url = 'https://www.estimize.com/calendar?page=%s&tab=equity&startDate=%s&endDate=%s' % (page_num, self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))
            elif announcement_type == 'Revenue':
                url = 'https://www.estimize.com/calendar?page=%s&tab=equity&metric=revenue&startDate=%s&endDate=%s' % (page_num, self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))

            self.driver.get(url)
            page_num = page_num + 1

            if max_page_num == 1000:
                try:
                    WebDriverWait(self.driver, self.delay).until(EC.presence_of_element_located((By.CLASS_NAME , 'eicEMB')))
                    elements = self.driver.find_elements_by_class_name('eicEMB')

                    for i in range(len(elements)):
                        if elements[i].text == 'Next':
                            max_page_num = int(elements[i-1].text)
                except:
                    pass

            if self.process_num==1:
                progress = (page_num-1)/float(max_page_num)
                print('Progress:', round(progress, 2), page_num-1, max_page_num, round(time.time()-self.start_time, 2))



            # method to extra the ticker symbols from the webpage
            tickers = self.get_tickers()
            try:
                df = pd.read_html(self.driver.page_source)[0]
            except Exception as e:
                return

            df['Symbol'] = tickers
            df = df.iloc[:, [2,3,5,6,8,9,10,12]]
            df.columns = ['Date Reported', 'Num of Estimates', 'Delta', 'Surprise', 'Wall St', 'Estimize', 'Actual', 'Symbol']

            date_reported_df = df['Date Reported'].str.split(' ', n = 1, expand = True)
            date_reported_df = date_reported_df.rename(columns={0:"Date Reported", 1:"Time Reported"})
            date_reported_df['Date Reported'] = pd.to_datetime(date_reported_df['Date Reported'])

            df['Date Reported'] = date_reported_df['Date Reported']
            df['Time Reported'] = date_reported_df['Time Reported']

            df.to_sql('estimize_%s' % announcement_type, self.conn, if_exists='append', index=False)
            first_ticker = self.get_first_ticker()
            if first_ticker == previous_first_ticker:
                print('Tickers are the same. Returning.')
                break

            previous_first_ticker = first_ticker

            if self.testing == True:
                break


    def get_combined_df(self):
        year = self.start_date.strftime('%y')

        eps_df = pd.read_sql('select * from estimize_EPS where "Date Reported" <= "%s" and "Date Reported" >= "%s"' % (self.end_date.strftime('%Y-%m-%d'), self.start_date.strftime('%Y-%m-%d')), self.conn)
        revenue_df = pd.read_sql('select * from estimize_Revenue where "Date Reported" <= "%s" and "Date Reported" >= "%s"' % (self.end_date.strftime('%Y-%m-%d'), self.start_date.strftime('%Y-%m-%d')), self.conn)


        eps_df = eps_df.sort_values(by='Date Reported')
        revenue_df = revenue_df.sort_values(by='Date Reported')

        eps_df = eps_df.set_index(['Date Reported', 'Symbol'], drop = True)
        revenue_df = revenue_df.set_index(['Date Reported', 'Symbol'], drop = True)

        eps_df.columns = 'EPS ' + eps_df.columns
        revenue_df.columns = 'Revenue ' + revenue_df.columns

        df = eps_df.join(revenue_df)

        return df


    def get_first_ticker(self):
        WebDriverWait(self.driver, self.delay).until(EC.presence_of_element_located((By.CLASS_NAME , 'lfkTWp')))
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

def drop_table(table_name):
    conn = sqlite3.connect('earnings.db', timeout=120)
    cur = conn.cursor()
    try:
        query = 'drop table estimize_%s' % table_name
        cur.execute(query)
        print('Dropped table', table_name)
    except Exception as e:
        print('Failed to drop table', table_name)
        pass


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
            drop_table('EPS')
            drop_table('Revenue')
            drop_table('data')

    lock = Lock()

    # create queue of dates
    date_queue = Queue()
    date_list = []

    if production == True:
        date_list.append((datetime.now(), datetime.now()))
    else:
        date_list.append((datetime.strptime('2013-01-01', '%Y-%m-%d'), datetime.strptime('2013-12-01', '%Y-%m-%d')))
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
        sleep(5)


    while date_queue.qsize()>0 or processes_running:
        sleep(15)
        #print(date_queue.qsize())

        if not any(process.is_alive() for process in processes):
            break
        else:
            print('process running')


    sleep(5)
