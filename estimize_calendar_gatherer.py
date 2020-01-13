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

# TODO: !!! Change to yearly instead of daily
# https://www.estimize.com/calendar?tab=equity&startDate=2019-01-01&endDate=2019-12-31

class calendar(Process):
    def __init__(self, date_queue, lock):
        Process.__init__(self)
        self.date_queue = date_queue
        self.lock = lock


    def run(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument('log-level=3')
        self.driver = webdriver.Chrome(options=chrome_options)
        self.delay = 4

        self.conn = sqlite3.connect('earnings.db', timeout=120)
        self.cur = self.conn.cursor()
        while self.date_queue.qsize()>0:
            self.read_date = self.date_queue.get()
            self.get_estimize_data()


    def get_estimize_data(self):
        # request the estimize website for data
        url = 'https://www.estimize.com/calendar?tab=equity&date=' + self.read_date.strftime('%Y-%m-%d')
        self.driver.get(url)

        # check if there are no companies reporting earnings
        myElem = WebDriverWait(self.driver, self.delay).until(EC.presence_of_element_located((By.CLASS_NAME , 'dAViVi')))
        companies_reporting_div = self.driver.find_element_by_class_name('dAViVi')
        if '0 Events' == companies_reporting_div.text.split('\n')[1]:
            return

        # check if earnings already in database
        symbol_href = self.driver.find_element_by_class_name('lfkTWp')
        symbol = symbol_href.text

        report_date = self.driver.find_element_by_class_name('dybmdC')
        report_date = report_date.text.replace('\n', '')

        try:
            # Check if already exists
            query = 'select * from estimize_eps where "Date Reported" == "%s"' % (report_date)
            self.cur.execute(query)
            results = self.cur.fetchall()
            if results != []:
                return
        except:
            pass

        # method to extra the ticker symbols from the webpage
        tickers = self.get_tickers()

        # method to get the historical data from yahoo
        #self.get_yahoo_historical(tickers)

        # read the table and make a dataframe out of it
        eps_df = pd.read_html(self.driver.page_source)[0]
        eps_df['Symbol'] = tickers

        # select only certain columns
        eps_df = eps_df.iloc[:, [2,3,5,6,7,8,9,10,12]]

        # rename columns
        eps_df.columns = ['Date Reported', 'Num of Estimates', 'Delta', 'Surprise', 'Historical Beat Rate', 'Wall St', 'Estimize', 'Actual', 'Symbol']

        # same as above, but for revenues table instead of EPS table
        url = 'https://www.estimize.com/calendar?tab=equity&metric=revenue&date=' + self.read_date.strftime('%Y-%m-%d')
        self.driver.get(url)
        myElem = WebDriverWait(self.driver, self.delay).until(EC.presence_of_element_located((By.TAG_NAME , 'table')))

        revenue_df = pd.read_html(self.driver.page_source)[0]
        tickers = self.get_tickers()
        revenue_df['Symbol'] = tickers
        revenue_df = revenue_df.iloc[:, [2,3,5,6,7,8,9,10,12]]
        revenue_df.columns = ['Date Reported', 'Num of Estimates', 'Delta', 'Surprise', 'Historical Beat Rate', 'Wall St', 'Estimize', 'Actual', 'Symbol']


        # store in the database
        eps_df.to_sql('estimize_eps', self.conn, if_exists='append', index=False)
        revenue_df.to_sql('estimize_revenue', self.conn, if_exists='append', index=False)


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
    lock = Lock()

    # create queue of dates
    date_queue = Queue()
    date_list = []
    read_date = datetime.strptime('2019-12-31', '%Y-%m-%d')
    # iterate through the dates
    while read_date > datetime.strptime('2013-01-01', '%Y-%m-%d'):
        date_list.append(read_date)

        read_date = read_date - timedelta(days=1)
        while read_date.weekday()>=5:      # exclude weekends
            read_date = read_date - timedelta(days=1)

    #shuffle(date_list)
    for date in date_list:
        date_queue.put(date)

    # start the program
    for i in range(7):
        p = calendar(date_queue, lock)
        p.start()
        sleep(1)

    while date_queue.qsize()>0:
        sleep(15)
        print(date_queue.qsize())
    sleep(60)
