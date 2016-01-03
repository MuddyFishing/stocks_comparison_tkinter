# -*- coding: utf-8 -*-
# -*- coding: iso-8859-15 -*-
#=======================================================#
# Python project for FE520                              #
#                            :Sandeep Joshi, Dec 2015   #
#=======================================================#
# Comparative Analysis of multiple stocks               #
#=======================================================#


__author__ = 'joshi'

import ystockquote
import datetime
import mysql.connector
import collections
import pandas as pd
from pandas import Series, DataFrame
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as legend
import sys
from statsmodels.tsa.stattools import acf, pacf
import holtwinters
from pandas.io.json import json_normalize
from stockbeta import calculateTrend
from Tkinter import *
import Tkinter
import ttkcalendar     # Not a standard package. Please see credits in the file
import tkSimpleDialog  # Not a standard package. Please see credits in the file
from ttk import Frame, Button, Style, Notebook
import ttk
import tkMessageBox
from ScrolledText import ScrolledText

welcome = """
===================================================
Welcome to Stock comparison tool by Sandeep Joshi
===================================================
C O N S O L E
+++++++++++++++++++++++++++++++++++++++++++++++++++
"""

TRAIN_INTERVAL = 365  # time interval in days
ROLLING_MEAN_WINDOW = 40
ROLLING_VAR_WINDOW = 30
DATE_FORMAT = "%Y-%m-%d"
CURRENT_DATE = datetime.datetime.now().strftime(DATE_FORMAT)
DD = datetime.timedelta(days=TRAIN_INTERVAL)
FROM_DATE = (datetime.datetime.now() - DD).strftime(DATE_FORMAT)
LAG = 1  # lag size for returns
PLOT_OPTIONS = ["Closing price", "Moving Average", "Logged Price", "Return (1 Day)", "Logged returns (1 day)",
                "Variance", "Logged variance",
                "ACF", "PACF", "ARIMA",
                "Combo plot 1 (Price/ returns)",
                "Combo plot 2 (Variance/ returns",
                "Combo plot 3 (Predictions)"]
COLUMNS = ("Beta", "Alpha", "r Squared", "Volatility", "Momentum")

COLUMNS_DAILY_KEY = ('fifty_two_week_low', 'fifty_day_moving_avg', 'price', 'price_book_ratio', 'volume',
                     'market_cap', 'dividend_yield', 'ebitda', 'change', 'dividend_per_share', 'stock_exchange',
                     'two_hundred_day_moving_avg', 'fifty_two_week_high', 'price_sales_ratio',
                     'price_earnings_growth_ratio', 'earnings_per_share', 'short_ratio', 'avg_daily_volume',
                     'price_earnings_ratio', 'book_value')

COLUMNS_DAILY = ("52 Week Low", "50 day mov avg", "Price", "Price book ratio", "Volume",
                 "Market cap", "Dividend yield", "EBITDA", "Change", "Dividend/ share", "Stock Exchg.",
                 "200 day mov avg", "52 week high", "Price sales ratio", "Price earning growth ratio",
                 "Earning/ share", "Short ratio", "Avg. daily volume", "Price earning ratio", "Book Volume")

IFACF = True  # Switch on/off acf

LINESTYLE = '--'
MARKER = 'o'
# SQL PARAMS
DB_DAILY = "stock_quote_daily"
DB_HISTORY = "stock_quote_historical"


def createDB():
    createHis = """
                   CREATE TABLE `stevens`.`{0}` (
                  `symbol` CHAR(6) NOT NULL,
                  `date` DATE NOT NULL,
                  `adj_close` FLOAT NULL,
                  `close` FLOAT NULL,
                  `high` FLOAT NULL,
                  `low` FLOAT NULL,
                  `open` FLOAT NULL,
                  `volume` INT(15) UNSIGNED NULL,
                   PRIMARY KEY (`symbol`, `date`));
                """.format(DB_HISTORY)

    createDaily = """
                   CREATE TABLE `stevens`.`{0}` (
                  `stock_exchange` CHAR(10) NOT NULL,
                  `symbol` CHAR(6) NOT NULL,
                  `date` DATE NOT NULL,
                  `avg_daily_vol` FLOAT NULL,
                  `book_value` FLOAT NULL,
                  `change` FLOAT NULL,
                  `dividend_per_share` FLOAT NULL,
                  `dividend_yield` FLOAT NULL,
                  `earnings_per_share` FLOAT NULL,
                  `ebitda` FLOAT NULL,
                  `fifty_day_mov_avg` FLOAT NULL,
                  `fifty_two_week_high` FLOAT NULL,
                  `fifty_two_week_low` FLOAT NULL,
                  `market_cap` FLOAT NULL,
                  `price` FLOAT NULL,
                  `price_book_ratio` FLOAT NULL,
                  `price_earn_growth_rate` FLOAT NULL,
                  `price_earn_ratio` FLOAT NULL,
                  `price_sales_ratio` FLOAT NULL,
                  `short_ratio` FLOAT NULL,
                  `two_hundred_day_mov_avg` FLOAT NULL,
                  `volume` INT(15) UNSIGNED NULL,
                   PRIMARY KEY (`stock_exchange`, `symbol`, `date`));
                  """.format(DB_DAILY)

    conn = connectToSql()
    cur = conn.cursor()
    cur.execute(createHis)
    cur.execute(createDaily)
    conn.commit()


# Get quotes from the internet using Yahoo API
def fetchQuotes(sym, start=FROM_DATE, end=CURRENT_DATE):
    his = None
    data = None
    try:
        # print start, end
        data = ystockquote.get_historical_prices(sym, start, end)
    except Exception:
        print "Please check the dates. Data might not be available. 404 returned"

        # 404 due to data yet not available
    if data:
        his = DataFrame(collections.OrderedDict(sorted(data.items()))).T
        his = his.convert_objects(convert_numeric=True)
        his.index = pd.to_datetime(his.index)
        his.insert(0, 'symbol', sym, allow_duplicates=True)
        # insert the date as dataframe too
        his.insert(1, 'date', his.index)
        # his.columns = getColumns('stock_quote_historical')   # Removing as db dependency is removed
        his.columns = getColumnsNoSql('stock_quote_historical')

    daily = ystockquote.get_all(sym)
    # print daily
    # persist(his, daily, sym, end)

    return his, daily


def getColumnsNoSql(dbtable):
    if dbtable=='stock_quote_historical':
        return [u'symbol', u'date', u'adj_close', u'close', u'high', u'low', u'open', u'volume']
    elif dbtable=='stock_quote_daily':
        return [u'stock_exchange', u'symbol', u'date', u'avg_daily_vol', u'book_value', u'change',
                u'dividend_per_share', u'dividend_yield', u'earnings_per_share', u'ebitda', u'fifty_day_mov_avg',
                u'fifty_two_week_high', u'fifty_two_week_low', u'market_cap', u'price', u'price_book_ratio',
                u'price_earn_growth_ratio', u'price_earn_ratio', u'price_sales_ratio', u'short_ratio',
                u'two_hundred_day_mov_avg', u'volume']

def getColumns(dbtable):
    conn = connectToSql()
    cur = conn.cursor()
    # print cur
    # Read
    query = ("""
    select column_name
      from information_schema.columns
     where table_schema = 'stevens'
       and table_name = '{0}';""".format(dbtable))

    cur.execute(query)
    return [item[0] for item in cur.fetchall()]


def toFloat(str):
    # there are B in end of some strings
    if str.endswith('B'):
        return float(str[:-1])
    elif str.endswith('N/A'):
        return 0.0
    else:
        return float(str)


def persist(his, daily, sym, end):
    # Save the daily numbers
    conn = connectToSql()
    cur = conn.cursor()
    # Insert
    query = ("""
    replace into stock_quote_daily
    values ('{0}', '{1}', '{2}', {3}, {4}, {5},
             {6}, {7}, {8}, {9}, {10}, {11}, {12},
            {13}, {14}, {15}, {16}, {17}, {18},
            {19}, {20}, {21});
            """.format(daily['stock_exchange'], sym, end,
                       daily['avg_daily_volume'],
                       daily['book_value'],
                       toFloat(daily['change']),
                       toFloat(daily['dividend_per_share']),
                       toFloat(daily['dividend_yield']),
                       toFloat(daily['earnings_per_share']),
                       toFloat(daily['ebitda']),
                       toFloat(daily['fifty_day_moving_avg']),
                       toFloat(daily['fifty_two_week_high']),
                       toFloat(daily['fifty_two_week_low']),
                       toFloat(daily['market_cap']),
                       toFloat(daily['price']),
                       toFloat(daily['price_book_ratio']),
                       toFloat(daily['price_earnings_growth_ratio']),
                       toFloat(daily['price_earnings_ratio']),
                       toFloat(daily['price_sales_ratio']),
                       toFloat(daily['short_ratio']),
                       toFloat(daily['two_hundred_day_moving_avg']),
                       toFloat(daily['volume'])))
    # print query
    cur.execute(query)

    # persist dataframe
    if not his.empty:
        his.to_sql(con=conn, name='stock_quote_historical', if_exists='replace', flavor='mysql', chunksize=10)
    conn.commit()


# Connects  to database
def connectToSql():
    cnxn = mysql.connector.connect(user='sandeep', password='iostream', host='localhost',
                                   database='stevens', buffered=True)
    return cnxn


# Get quotes/ checks locally if not found then gets from net
def getQuotes(sym, start_date=FROM_DATE, end_date=CURRENT_DATE):
    # look for the quotes in the database
    conn = connectToSql()
    cur = conn.cursor()
    # Check all the dates that are available in loco
    query = ("""
    select * from stock_quote_historical
     where symbol = '{0}'
       and date >= '{1}'
       and date <= '{2}';
       """.format(sym, start_date, end_date))
    # print query
    df_history = pd.read_sql(query, con=conn)

    # Check if both the dates are in there
    if len(df_history) == 0:
        his, daily = fetchQuotes(sym, start_date, end_date)
    else:
        # get data between the latest day in dataframe and current date
        df_history = df_history.sort_index(axis=1, ascending=False)
        # print(df_history.index[0][1])
        if df_history.index[0][1] != end_date:
            his, daily = fetchQuotes(sym, df_history.index[0][1], end_date)
        else:
            # Get the daily quotes too
            query = ("""
            select * from stock_quote_daily
             where symbol = '{0}'
              and date = '{1}';
              """.format(sym, end_date))
            # print query
            daily = cur.execute(query).fetchone()

    conn.close()
    if not df_history.empty:
        if not his.empty:
            df_history.append(his)
    else:
        df_history = his
    return df_history, daily


def chart(plt, id, subplot, title, xlab, ylab, vals, label, ifPred=False):
    plt.figure(id)
    plt.subplot(subplot)
    plt.interactive(False)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title)

    if ifPred:
        return plt.plot(vals, label=label, marker=MARKER, linestyle=LINESTYLE)
    return plt.plot(vals, label=label)


def test_model(name, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)  # train the model
    y_test[name] = model.predict(x_test)  # test the model
    # Verify


def calc_vol_ret(data):
    colname = data.columns[2]
    # data.dropna()
    # Create gap here if more than 1 day is required by shifting data
    data['return'] = data[colname] - data[colname].shift(LAG)
    # Natural log
    data['nlog'] = data[colname].apply(lambda f: np.log(f))
    # Logged Difference
    # print(data.head())
    data['lreturn'] = data['nlog'] - data['nlog'].shift(LAG)
    # Variance
    data['var'] = pd.rolling_var(data[colname], ROLLING_VAR_WINDOW, min_periods=None, freq=None, center=True)
    # Logged Variance
    data['logvar'] = pd.rolling_var(data['nlog'], ROLLING_VAR_WINDOW, min_periods=None, freq=None, center=True)

    if IFACF:
        acfRes, pacfRes = getacf(data)
        return acfRes, pacfRes, data
    else:
        return data


def getacf(data):
    temp = data['lreturn'].iloc[1:]
    acfRes = acf(temp)
    pacfRes = pacf(temp)
    return acfRes, pacfRes


def arima(data):
    model = sm.tsa.ARIMA(data['nlog'].iloc[1:], order=(1, 0, 0))
    result = model.fit(disp=-1)
    return result.fittedvalues


def fetch_data(ticker, fr_date, to_date):
    cell_text = []
    # Fetch and Load data
    data = {}
    for sym in ticker:
        temp = {}
        print 'processing: ', sym
        # temp['history'], temp['daily'] = getQuotes(sym, fr_date, to_date) # commented to avoid SQL
        if not fr_date and not to_date:
            temp['history'], temp['daily'] = fetchQuotes(sym)
        else:
            temp['history'], temp['daily'] = fetchQuotes(sym, fr_date, to_date)
        temp['history'].convert_objects(convert_numeric=True)
        # Calculate moving avg
        temp['history']['mean'] = pd.rolling_mean(temp['history'].reindex(columns=[temp['history'].columns[3]]),
                                                  ROLLING_MEAN_WINDOW)

        # Calculate volatility and returns
        rev_data = pd.DataFrame(temp['history'][::-1], index=temp['history'].index[::-1])
        # print rev_data.head()
        if IFACF:
            acfRes, pcfRes, rev_data = calc_vol_ret(rev_data)
        else:
            rev_data = calc_vol_ret(rev_data)

        # ARIMA forecast based on Natural log
        pred = arima(rev_data)

        data[sym] = {'history': rev_data, 'daily': temp['daily'], 'acf': acfRes, 'pcf': pcfRes, 'pred_arima': pred}
    return data


class CalendarDialog(tkSimpleDialog.Dialog):
    """Dialog box that displays a calendar and returns the selected date"""
    def body(self, master):
        self.calendar = ttkcalendar.Calendar(master)
        self.calendar.pack()

    def apply(self):
        self.result = self.calendar.selection


class CalendarFrame(Tkinter.LabelFrame):
    def __init__(self, master, text, default):
        Tkinter.LabelFrame.__init__(self, master, text=text)

        def getdate():
            cd = CalendarDialog(self)
            result = cd.result
            self.selected_date.set(result.strftime(DATE_FORMAT))

        self.selected_date = Tkinter.StringVar()

        Tkinter.Entry(self, textvariable=self.selected_date).pack(side=Tkinter.LEFT)
        Tkinter.Button(self, text="...", command=getdate).pack(side=Tkinter.LEFT)

    def getDate(self):
        return self.selected_date.get()


class App:

    def __init__(self, master):
        master.minsize(width=1500, height=1500)
        root = Frame(master)
        root.pack(side=TOP, fill=BOTH)
        master.title('Stocks evaluation - Sandeep Joshi (FE520)')

        n = Notebook(root)
        n.enable_traversal()
        n.pack(fill=BOTH)
        self._create_stat_tab(n)
        self._create_help_tab(n)
        self.data = {}
        self.start = ''
        self.end = ''
        self.tickers = []
        self.tree

    def _create_help_tab(self, n):
        #==================================
        # Help tab

        layer_b1 = Frame(n)
        layer_b1.pack(side=TOP)

        explain = [
            """
            β: measure of relative risk of the stock with respect to the market (S&P500)\n.
            α: measure of the excess return with respect to the benchmark.

                Eq: R_i^stock="α"+"β x " R_i^market+ε_i

            R2: measure of how well the the returns of a stock is explained by the returns of the benchmark.

            Volatility: Standard deviation of the returned stocks

            Momentum: measure of the past returns over a certain period of time.


            More details @ http://gouthamanbalaraman.com/blog/calculating-stock-beta.html
            """
        ]

        definition = LabelFrame(layer_b1, text="Definitions:")
        definition.pack(side=TOP, fill=BOTH)
        lbl = Label(definition, wraplength='10i', justify=LEFT, anchor=N, text=''.join(explain))
        lbl.pack(anchor=NW)

        msg = [
            """
            Created by Sandeep Joshi for class FE520

            Under guidance of Prof. Peter Lin @Stevens 2015


            Special thanks to following people/ resources on the net:

            •	Sentdex @Youtube
            •	http://gouthamanbalaraman.com/blog/calculating-stock-beta.html
            •	http://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/
            •	http://francescopochetti.com/category/stock-project/
            •	Moshe from SO -> http://stackoverflow.com/users/875832/moshe
            """
        ]
        credits = LabelFrame(layer_b1, text="Credits:")
        credits.pack(side=TOP, fill=BOTH)

        lbl = Label(credits, wraplength='10i', justify=LEFT, anchor=S, text=''.join(msg))
        lbl.pack(anchor=NW)
        n.add(layer_b1, text="Help/ Credits", underline=0, padding=2)

    def _create_stat_tab(self, n):
        layer1 = Frame(n)   # first page, which would get widgets gridded into it

        #===================================
        # Layer 1 (Present day present time)
        layer1.pack(side=TOP)
        tickers = []

        layer1_0 = Frame(layer1)
        layer1_0.pack(side=TOP, anchor=W)

        # Ticker input
        self.ticker = StringVar()
        self.ticker.set("GOOG, MSFT")
        # self.from_dt = StringVar()
        # self.to_dt = StringVar()
        # self.from_dt.set(FROM_DATE)
        # self.to_dt.set(CURRENT_DATE)
        layer1_1 = LabelFrame(layer1_0, text="Tickers")
        layer1_1.pack(side=LEFT, anchor=W)

        ticker = Entry(layer1_1, textvariable=self.ticker, bd=5)
        ticker.pack(side=LEFT, anchor=W)

        # Date range
        self.from_dt = CalendarFrame(layer1_0, "From", FROM_DATE)
        self.to_dt = CalendarFrame(layer1_0, "To", CURRENT_DATE)

        fetch = Button(layer1_1, text="Get Data", command=self.fetch)
        fetch.pack(side=LEFT)
        self.from_dt.pack(side=LEFT)
        self.to_dt.pack(side=LEFT)

        #===================================
        # Layer 2  Two panes side-by-side
        # Add panes
        pane_plot = PanedWindow(layer1, orient=HORIZONTAL)
        pane_plot.pack(side=TOP, fill=BOTH, expand=1)

        left = LabelFrame(pane_plot, text="Plotting options")
        pane_plot.add(left)

        right = LabelFrame(pane_plot, text="Trends")
        pane_plot.add(right)

        # Left Pane
        # Plotting options
        scrollText = Frame(left)
        scrollText.pack()
        scrollbar = Scrollbar(scrollText, orient=VERTICAL)
        self.plot_op = Listbox(scrollText, selectmode=EXTENDED, yscrollcommand=scrollbar.set)
        self.plot_op.pack(side=LEFT, anchor=NE)
        scrollbar.config(command=self.plot_op.yview)
        scrollbar.pack(side=LEFT, fill=Y)
        for item in PLOT_OPTIONS:
            self.plot_op.insert(END, item)
        self.plot = Button(left, text="Plot", command=self.plot)
        self.plot.pack(anchor=W)

        # Right pane
        # Historical data coming from stockbeta.py
        frame_hist = LabelFrame(right, text='Historical data', width=80)
        frame_hist.pack(side=TOP, anchor=NW)
        self.tree = ttk.Treeview(frame_hist)
        # style = ttk.Style(frame_hist)
        # style.configure('Treeview', rowheight=8, columnwidth=10)
        ysb = ttk.Scrollbar(frame_hist, orient='vertical', command=self.tree.yview)
        xsb = ttk.Scrollbar(frame_hist, orient='horizontal', command=self.tree.xview)
        xsb.pack(fill=X)
        self.tree.configure(yscroll=ysb.set, xscroll=xsb.set)
        self.tree["columns"] = COLUMNS
        for param in COLUMNS:
            self.tree.column(param, width=100)
            self.tree.heading(param, text=param)
        ysb.pack(side=LEFT, fill=Y)
        self.tree.pack(side=LEFT)

        # Current stock data
        frame_daily = LabelFrame(right, text="Today's data", width=80)
        # frame_daily.pack_propagate(0)
        frame_daily.pack(side=TOP)
        self.tree2 = ttk.Treeview(frame_daily)
        ysb2 = ttk.Scrollbar(frame_daily, orient='vertical', command=self.tree2.yview)
        xsb2 = ttk.Scrollbar(frame_daily, orient='horizontal', command=self.tree2.xview)
        xsb2.pack(fill=X)
        self.tree2.configure(yscroll=ysb2.set, xscroll=xsb2.set)
        self.tree2["columns"] = COLUMNS_DAILY
        for param in COLUMNS_DAILY:
            self.tree2.column(param, width=100)
            self.tree2.heading(param, text=param)

        ysb2.pack(side=LEFT, fill=Y)
        self.tree2.pack(side=LEFT)

        closebar = Frame(right, relief=RAISED, borderwidth=1)
        closebar.pack(anchor=E, expand=True)
        self.console = ScrolledText(closebar)
        self.console = ScrolledText(bg='gray', height=7)
        # self.console.config(state=DISABLED)
        self.console['font'] = ('consolas', '12')
        self.console.pack(expand=True)
        self.console.insert(INSERT, welcome+'\n')
        button = Button(closebar, text="QUIT", command=layer1.quit)
        button.pack(anchor=SE, expand=True)
        n.add(layer1, text='Stats')

    def about(self):
        temp = Tk()
        w = Message(temp, text="Created by Sandeep Joshi for FE520 @Stevens.edu", width=500)
        w.pack()

    def plot(self):
        def addLegend(plot_type, plot, legendlist):
            if plot_type in legendlist:
                legendlist[plot_type].append(plot)
            else:
                legendlist[plot_type] = [plot]

        if len(self.plot_op.curselection()) < 1:
            msg = "Please choose a Plot type in the left pane."
            tkMessageBox.showinfo("Error", msg)
            self.console.insert(INSERT, msg+'\n')
            pass

        i = 0
        if not self.data:
            msg = "Please Get data first before making plots."
            tkMessageBox.showinfo("Error", msg)
            self.console.insert(INSERT, msg+'\n')
            pass

        map = {
            '0': 3,  # Closing price
            '1': 8,  # Moving average
            '2': 10,  # Logged price
            '3': 9,  # return
            '4': 11,  # Logged return
            '5': 12,  # Variance
            '6': 13,  # Logged Variance
        }

        legends = dict()
        # Loop at the data
        for key, value in self.data.iteritems():
            print key
            i += 1
            for key2, value2 in value.iteritems():
                pos_id = len(self.data)*100 + 10 + i  # deduces plots for combo options
                # Loop the plot selections
                for item in self.plot_op.curselection():

                    if key2 == 'history':
                        self.console.insert(END, "Drawing {0} for {1}...".format(PLOT_OPTIONS[item], key)+'\n')
                        print "Drawing {0} for {1}...".format(PLOT_OPTIONS[item], key)
                        if item < 7:
                            l1, = chart(plt, PLOT_OPTIONS[item], 111, PLOT_OPTIONS[item], 'Date', 'Price',
                                  value2[value2.columns[map[str(item)]]], key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)
                            continue

                        if item == 10:
                            l1, = chart(plt, PLOT_OPTIONS[item], 311, 'Closing Price', 'Date', 'Price',
                                        value2[value2.columns[3]], key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)

                            l1, = chart(plt, PLOT_OPTIONS[item], 312, 'Moving Avg.', 'Date', 'Price', value2['mean'],
                                        key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)
                            l1, = chart(plt, PLOT_OPTIONS[item], 313, 'Returns for LAG {0}'.format(str(LAG)), 'Date',
                                        'Price', value2['return'], key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)
                            continue

                        if item == 11:
                            l1, = chart(plt, PLOT_OPTIONS[item], 311, 'Variance', 'Date', 'Price', value2['var'], key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)
                            l1, = chart(plt, PLOT_OPTIONS[item], 312, 'Log Variance', 'Date', 'Price', value2['logvar'],
                                        key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)
                            l1, = chart(plt, PLOT_OPTIONS[item], 313, 'Logged first diff', 'Date', 'Price',
                                        value2['lreturn'], key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)
                            continue

                    elif key2 == 'acf':
                        if item == 7:
                            l1, = chart(plt, PLOT_OPTIONS[item], 111, PLOT_OPTIONS[item], 'Date', 'Price', value2, key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)
                        elif item == 12:
                            l1, = chart(plt, PLOT_OPTIONS[item], 311, PLOT_OPTIONS[item], 'Date', 'Price', value2, key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)


                    elif key2 == 'pcf':
                        if item == 8:
                            l1, = chart(plt, PLOT_OPTIONS[item], 111, PLOT_OPTIONS[item], 'Date', 'Price', value2, key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)
                        elif item == 12:
                            l1, = chart(plt, PLOT_OPTIONS[item], 311, PLOT_OPTIONS[item], 'Date', 'Price', value2, key)


                    elif key2 == 'pred_arima':
                        if item == 9:
                            l1, = chart(plt, PLOT_OPTIONS[item], 111, PLOT_OPTIONS[item], 'Date', 'Price', value2, key)
                            addLegend(PLOT_OPTIONS[item], l1, legends)

                        if item == 12:  # Preditions combo
                            chart(plt, PLOT_OPTIONS[item], 312, 'ARIMA', 'Date', 'Price', value2, key,
                                  True)
                            chart(plt, PLOT_OPTIONS[item], 312, 'Log', 'Date', 'Price',
                                  self.data[key]['history']['nlog'], key)

                            chart(plt, PLOT_OPTIONS[item], 313, 'ARIMA', 'Date', 'Price', value2, key, True)
                            chart(plt, PLOT_OPTIONS[item], 313, 'Log First Difference', 'Date', 'Price',
                                  self.data[key]['history']['lreturn'], key)

        for item in self.plot_op.curselection():
            # print 'legend for', legends[PLOT_OPTIONS[item]]
            try:
                print legends[PLOT_OPTIONS[item]]
                # if item > 9:
                #     print self.tickers
                #     plt.figlegend(lines=legends[PLOT_OPTIONS[item]], labels=self.tickers, loc='lower center')
                # else:
                plt.legend(handles=legends[PLOT_OPTIONS[item]], loc='upper left')
            except KeyError:
                continue
        plt.show()

    def fetch(self):
        # read the variable for ticker and populate the list
        if not self.ticker:
            msg = "Please Enter tickers separated by ','."
            tkMessageBox.showinfo("Error", msg)
            self.console.insert(END, msg+'\n')
            pass
        self.end = self.to_dt.getDate()
        self.start = self.from_dt.getDate()
        self.tickers = self.ticker.get().replace(" ", "").split(',')
        self.data = fetch_data(self.tickers, self.start, self.end)

        # Delete the tree
        self.tree.delete(*self.tree.get_children())
        self.tree2.delete(*self.tree2.get_children())
        # Populate the tree
        for sym in self.tickers:
            values = []
            values.append([x for x in calculateTrend(sym, self.end)])
            self.tree.insert("", 0, text=sym, values=tuple(values[0]))
            values2 = []
            for item in COLUMNS_DAILY_KEY:
                values2.append(self.data[sym]['daily'][item])
            self.tree2.insert("", 0, text=sym, values=tuple(values2))


    def explain(self):
        # Explain the alpha, beta
        pop_up = Tk()
        help = App(pop_up)


def main(argv):

    root = Tk()
    app = App(root)
    root.mainloop()
    root.destroy()

    # Create database if not there
    # createDB()
    # Companies

    # Plot
    # i = 0
    # line = []
    # cols = getColumns('stock_quote_historical')
    # print cols
    # for key, value in data.iteritems():
    #     i+=1
    #     for key2, value2 in value.iteritems():
    #         pos_id = len(data)*100 + 10 + i
    #         if key2 == 'history':
    #             # For graphs on same chart
    #
    #             chart(plt, 'Historical comparison 1', 311, 'Closing Price', 'Date', 'Price', value2[value2.columns[3]],
    #                   sym)
    #             # plt.legend(stock_symbols, handles=[line], loc='upper left')
    #             chart(plt, 'Historical comparison 1', 312, 'Moving Avg.', 'Date', 'Price', value2['mean'], sym)
    #             # plt.legend(stock_symbols, loc='upper left')
    #             chart(plt, 'Historical comparison 1', 313, 'Returns for LAG {0}'.format(str(LAG)), 'Date', 'Price',
    #                   value2['return'], sym)
    #
    #             chart(plt, 'Historical comparison 2', 311, 'Variance', 'Date', 'Price', value2['var'], sym)
    #             chart(plt, 'Historical comparison 2', 312, 'Log Variance', 'Date', 'Price', value2['logvar'], sym)
    #             chart(plt, 'Historical comparison 2', 313, 'Logged first diff', 'Date', 'Price', value2['lreturn'], sym)
    #
    #         elif key2 == 'acf':
    #              chart(plt, 'Fit/Prediction - {0}'.format(sym), 311, 'ACF', 'Date', 'Price', value2, sym, True)
    #
    #         elif key2 == 'pcf':
    #             chart(plt, 'Fit/Prediction - {0}'.format(sym), 311, 'PCF', 'Date', 'Price', value2, sym)
    #
    #         elif key2 == 'pred_arima':
    #             chart(plt, 'Fit/Prediction - {0}'.format(sym), 312, 'ARIMA', 'Date', 'Price', value2, sym, True)
    #             chart(plt, 'Fit/Prediction - {0}'.format(sym), 312, 'Log', 'Date', 'Price', data[sym]['history']['nlog']
    #                   , sym)
    #
    #             chart(plt, 'Fit/Prediction - {0}'.format(sym), 313, 'ARIMA', 'Date', 'Price', value2, sym, True)
    #             chart(plt, 'Fit/Prediction - {0}'.format(sym), 313, 'Log First Difference', 'Date', 'Price',
    #                   data[sym]['history']['lreturn'], sym)
    #
    #
    #     # Get last five year trends for the stocks
    #     print key
    #     print calculateTrend(key, CURRENT_DATE)
    #     cell_text.append([x for x in calculateTrend(key, CURRENT_DATE)])
    #
    # # Table
    # rows = stock_symbols
    # plt.figure('Trends')
    # columns = COLUMNS[1:]
    # table = plt.table(cellText=cell_text,
    #                   colWidths=[.05]*5,
    #                   rowLabels=rows,
    #                   # rowColours=,
    #                   colLabels=columns,
    #                   loc='center')
    # table.auto_set_font_size(False)
    # plt.text(20,3.4,'Trends of last 5 years', size=18)
    # table.set_fontsize(20)
    # table.scale(4,4)
    # plt.plot()

    # legend(loc=2)
    # legend(line, stock_symbols)

    # handles, labels = plt.get_legend_handles_labels()
    # sort both labels and handles by labels
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # plt.legend(handles, labels)


    # Output results and graphs

    # plt.show()

if __name__ == "__main__":
    main(sys.argv)

