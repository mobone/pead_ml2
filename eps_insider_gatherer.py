from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

import pandas as pd

#driver = webdriver.Firefox(executable_path = r"C:\Users\nbrei\Documents\GitHub\quarterly_earnings\geckodriver.exe")

driver = webdriver.Firefox(executable_path = r"C:\Users\nbrei\Documents\GitHub\quarterly_earnings\geckodriver.exe")
delay = 2
class company():
    def __init__(self, symbol):
        self.symbol = symbol
        self.get_page_source()

    def get_page_source(self):
        for metric_name in [('revenue', 'Revenue'), ('eps','EPS')]:
            print('here', metric_name)
            url = 'https://www.estimize.com/{}/fq4-2018?metric_name={}&chart=table'.format(self.symbol, metric_name[0])
            driver.get(url)

            myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME , 'rel-chart-tbl')))
            df = pd.read_html(driver.page_source)[0]
            df = df.iloc[[1,2,5,8,11,12]]

            df = df.set_index(metric_name[1])
            print(df)
            res = pd.melt(df.reset_index(), id_vars=[metric_name[1]])
            print(res)
            res['col_name'] = res[metric_name[1]] + ' ' + res['variable']
            print(res)



company('BBY')
