import sqlite3


class announcement:
    def __init__(eps_df, revenue_df):
        conn = sqlite3.connect("earnings.db", timeout=120)
        cur = conn.cursor()

        symbol_href = self.driver.find_element_by_class_name("lfkTWp")
        symbol = symbol_href.text

        eps_history_df = pd.read_sql(
            'select * from estimize_eps where Symbol == "%s"' % symbol, conn
        )
        revenue_history_df = pd.read_sql("select * from estimize_revenue", conn)
        price_history_df = pd.read_sql("select * from price_history", conn)

    def get_combined_df(eps_df, revenue_df):
        del eps_df["Historical Beat Rate"]
        del revenue_df["Historical Beat Rate"]

        date_reported_df = eps_df["Date Reported"].str.split(" ", n=1, expand=True)
        date_reported_df = date_reported_df.rename(
            columns={0: "Date Reported", 1: "Time Reported"}
        )
        date_reported_df["Date Reported"] = pd.to_datetime(
            date_reported_df["Date Reported"]
        )
        eps_df["Date Reported"] = date_reported_df["Date Reported"]
        eps_df["Time Reported"] = date_reported_df["Time Reported"]

        date_reported_df = revenue_df["Date Reported"].str.split(" ", n=1, expand=True)
        date_reported_df = date_reported_df.rename(
            columns={0: "Date Reported", 1: "Time Reported"}
        )
        date_reported_df["Date Reported"] = pd.to_datetime(
            date_reported_df["Date Reported"]
        )
        revenue_df["Date Reported"] = date_reported_df["Date Reported"]
        revenue_df["Time Reported"] = date_reported_df["Time Reported"]

        eps_df = eps_df.sort_values(by="Date Reported")
        revenue_df = revenue_df.sort_values(by="Date Reported")

        eps_df = eps_df.set_index(
            ["Date Reported", "Time Reported", "Symbol"], append=True, drop=True
        )
        revenue_df = revenue_df.set_index(
            ["Date Reported", "Time Reported", "Symbol"], append=True, drop=True
        )

        eps_df.columns = "EPS " + eps_df.columns
        revenue_df.columns = "Revenue " + revenue_df.columns

        df = eps_df.join(revenue_df)

        return df

    def get_historical_beat():
        df["Historical EPS Beat Ratio"] = None
        df["Historical EPS Beat Percent"] = None
        for index, row in df.iterrows():
            index_num, date_reported, time_reported, symbol = index

            this_df = df[df.index.get_level_values("Symbol") == symbol]
            beat_rate = this_df[
                this_df.index.get_level_values("Date Reported") <= date_reported
            ].tail(8)

            if len(beat_rate) >= 4:
                beat_rate_ratio = len(beat_rate[beat_rate["EPS Surprise"] > 0]) / float(
                    len(beat_rate)
                )
                beat_rate_percent = beat_rate["EPS Surprise"] / beat_rate["EPS Actual"]
                beat_rate_percent = beat_rate_percent.replace([np.inf, -np.inf], np.nan)
                beat_rate_percent = beat_rate_percent.mean()

                # TODO: Do the same for revenue
                df.loc[index_num, ["Historical EPS Beat Ratio"]] = beat_rate_ratio
                df.loc[index_num, ["Historical EPS Beat Percent"]] = beat_rate_percent

    def get_average_change():
        df["Average Change 5 Days"] = None
        df["Average Abnormal Change 5 Days"] = None
        df["Average Change 10 Days"] = None
        df["Average Abnormal Change 10 Days"] = None
        for index, row in df.iterrows():
            index_num, date_reported, time_reported, symbol = index

            returns_df = df[
                df.index.get_level_values("Date Reported") < date_reported
            ].tail(8)

            if len(returns_df) >= 4:
                df.loc[index_num, ["Average Change 5 Days"]] = returns_df[
                    "5 Day Change"
                ].mean()
                df.loc[index_num, ["Average Change 10 Days"]] = returns_df[
                    "10 Day Change"
                ].mean()
                df.loc[index_num, ["Average Abnormal Change 5 Days"]] = returns_df[
                    "5 Day Change Abnormal"
                ].mean()
                df.loc[index_num, ["Average Abnormal Change 10 Days"]] = returns_df[
                    "10 Day Change Abnormal"
                ].mean()

    def get_YoY_growth():
        df["YoY Growth"] = None
        for index, row in df.iterrows():
            index_num, date_reported, time_reported, symbol = index
            time_reported = time_reported.replace("'", "")
            quarter_numer, year = time_reported.split(" ")

            this_df = df["EPS Actual"]
            try:
                this_quarter = this_df[
                    this_df.index.get_level_values("Time Reported")
                    == quarter_numer + " '" + year
                ].values[0]
                last_quarter = this_df[
                    this_df.index.get_level_values("Time Reported")
                    == quarter_numer + " '" + str(int(year) - 1)
                ].values[0]
                df.loc[index_num, ["YoY Growth"]] = (
                    this_quarter - last_quarter
                ) / last_quarter
            except Exception as e:
                pass

    def get_market_cap():
        finviz_page = r.get("https://finviz.com/quote.ashx?t=%s" % symbol)

        soup = BeautifulSoup(finviz_page.text, features="lxml")
        table_row = soup.findAll("tr", attrs={"class": "table-dark-row"})[1]
        market_cap = table_row.text.replace("Market Cap", "").split("\n")[1]
        if "K" in market_cap:
            market_cap = float(market_cap[:-1]) * 1000
        elif "M" in market_cap:
            market_cap = float(market_cap[:-1]) * 1000000
        elif "B" in market_cap:
            market_cap = float(market_cap[:-1]) * 1000000000

        market_cap = int(market_cap)
        if market_cap > 10000000000:
            market_cap_text = "Large"
        elif market_cap > 2000000000:
            market_cap_text = "Medium"
        elif market_cap > 300000000:
            market_cap_text = "Small"
        elif market_cap > 50000000:
            market_cap_text = "Micro"
        else:
            market_cap_text = "Nano"

        df["Market Cap Text"] = market_cap_text


def get_estimize_data(self):
    # request the estimize website for data
    url = "https://www.estimize.com/calendar?tab=equity&date=" + datetime.now().strftime(
        "%Y-%m-%d"
    )
    self.driver.get(url)

    # check if there are no companies reporting earnings
    myElem = WebDriverWait(self.driver, self.delay).until(
        EC.presence_of_element_located((By.CLASS_NAME, "dAViVi"))
    )
    companies_reporting_div = self.driver.find_element_by_class_name("dAViVi")
    if "0 Events" == companies_reporting_div.text.split("\n")[1]:
        return

    # method to extra the ticker symbols from the webpage
    tickers = self.get_tickers()

    # method to get the historical data from yahoo
    # self.get_yahoo_historical(tickers)
    # TODO: update price history table with missing yahoo price data entries

    # read the table and make a dataframe out of it
    eps_df = pd.read_html(self.driver.page_source)[0]
    eps_df["Symbol"] = tickers
    eps_df = eps_df.iloc[:, [2, 3, 5, 6, 7, 8, 9, 10, 12]]
    eps_df.columns = [
        "Date Reported",
        "Num of Estimates",
        "Delta",
        "Surprise",
        "Historical Beat Rate",
        "Wall St",
        "Estimize",
        "Actual",
        "Symbol",
    ]

    # same as above, but for revenues table instead of EPS table
    url = (
        "https://www.estimize.com/calendar?tab=equity&metric=revenue&date="
        + self.read_date.strftime("%Y-%m-%d")
    )
    self.driver.get(url)
    myElem = WebDriverWait(self.driver, self.delay).until(
        EC.presence_of_element_located((By.TAG_NAME, "table"))
    )

    revenue_df = pd.read_html(self.driver.page_source)[0]
    tickers = self.get_tickers()
    revenue_df["Symbol"] = tickers
    revenue_df = revenue_df.iloc[:, [2, 3, 5, 6, 7, 8, 9, 10, 12]]
    revenue_df.columns = [
        "Date Reported",
        "Num of Estimates",
        "Delta",
        "Surprise",
        "Historical Beat Rate",
        "Wall St",
        "Estimize",
        "Actual",
        "Symbol",
    ]

    return eps_df, revenue_df


def get_tickers(self):
    # extract ticker symbopls from the html source
    soup = BeautifulSoup(self.driver.page_source, features="lxml")
    ticker_links = soup.findAll("a", attrs={"class": "lfkTWp"})

    # create list of symbols that were extracted
    tickers = []
    for ticker in ticker_links:
        tickers.append(ticker.contents[0])

    return tickers
