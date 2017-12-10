##Dependencies
from collections import Counter                                         #counter allows us to deal with strings
import datetime as dt
import os   #can be used to create new directories
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc                         #requires mdate, ohlc
import matplotlib.dates as mdates                                       #Matplotlib doesn't use datetime dates
import pandas as pd                                                     #Data analysis library 
import pandas_datareader.data as web                                    #Grabs data from Yahoo Finance API
import numpy as np 
import sklearn
import bs4 as bs                                                        #beautiful soup
import pickle                                                           #serializes any python object (saves any object)
import requests
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


style.use('ggplot')                                                     #Just one of many matplot styles you can use

##Write Tesla stock info to csv
#start = dt.datetime(2000,1,1)
#end = dt.datetime(2016,12,31)
#df = web.DataReader('TSLA', 'yahoo', start, end)                       #Gets pandas dataframe (spreadsheet) from yahoo finance for Tesla start and end time
#df.to_csv('tsla.csv')

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)


##Print the first few days of opening and high prices 
#print(df.head())
#print(df[['Open','High']].head())


##Plot Adjusted Closing prices
#df['Adj Close'].plot()
#plt.show()


##Create 100-day moving average
#df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
#df.dropna(inplace=True)  #removes all NAN entries --aka-- first 100 days
#print(df.tail())


##Plotting using just matplotlib
#ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)  #mxn, start point, rowspan, colspan
#ax2 = plt.subplot2grid((6,1), (5,0), rowspan=5, colspan=1, sharex=ax1)
#ax1.plot(df.index, df['Adj Close'])                                                                      #plots line, taking date as 'x' and adj close as 'y'
#ax1.plot(df.index, df['100ma']) 
#ax2.bar(df.index, df['Volume'])                                                                          #bar plot, date against volume
#plt.show()


##Resampling with pandas (eg, resampling stock tick data to 1-day or 10-day)
#ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)  #mxn, start point, rowspan, colspan
#ax2 = plt.subplot2grid((6,1), (5,0), rowspan=5, colspan=1, sharex=ax1)
#df_ohlc = df['Adj Close'].resample('10D').ohlc()                                                          #Resamples adj close based on every 10 days of data and outputs only open/high/low/close (can also do .sum() or .mean())
#df_volume = df['Volume'].resample('10D').sum()                                                            #Resamples sum of volume data over 10 days


##Plot candlestick graph
#df_ohlc.reset_index(inplace=True)   #resets date to be a column
#df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)    #converts datetime object to mdate 
#ax1.xaxis_date()   #displays mdates as beautiful dates
#candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
#ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
#plt.show()


##Grab S&P500 Tickers (top 500 companies by market cap)
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')                     #uses requests to get S&P500 list wiki page
    soup = bs.BeautifulSoup(resp.text, 'lxml')                                                           #converts text of source code to beautiful soup, uses lxml parser
    table = soup.find('table', {'class':'wikitable sortable'})                                           #find the S&P500 wikitable, specifying class and class type 
    tickers = []
    for row in table.findAll('tr')[1:]:                                                                  #for each table row from 1 onward
        ticker = row.findAll('td')[0].text                                                               #just select the 0th column (ticker), convert from soup object to text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:                                                         #saves as pickle, write bytes as f
        pickle.dump(tickers, f)                                                                          #dump tickers to file f
    print(tickers)
    return tickers
#save_sp500_tickers()


##Grab S&P500 Company Data from Yahoo Finance API
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()                                                                   #if someone wants to reload sp500 tickers, will rerun save_sp500 function
    else:
        with open("sp500tickers.pickle", "rb") as f:                                                     #else loads S&P500 ticker pickle 
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000,1,1)
    end = dt.datetime(2016,12,31)

    for ticker in tickers:
         print(ticker)
         if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)                                             #Read from Yahoo for whatever ticker is
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
         else:
            print('Already have {}'.format(ticker))
#get_data_from_yahoo()


##Compile separate company DataFrames into one large DataFrame
def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()                                                                             #initialize new empty pandas DataFrame

    for count, ticker in enumerate(tickers):                                                             #iterate through latest tickers that we have (returns count and nth element)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            continue
        else:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date',inplace=True)
            df.rename(columns = {'Adj Close': ticker}, inplace=True)                                     #rename adj close column to be ticker name

        df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)                                 #remove everything but ticker (adj close) column
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')                                                      #joins dataframes, without dropping non-redundancies

        if count % 10 == 0:                                                                              #if count divisible by 10 remainder 0, then print count to keep track of where we are
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')
#compile_data()


##Find and Visualize Relationships (Correlation) between S&P500 Companies
def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    #df['TSLA'].plot()
    #plt.show()
    df_corr = df.corr()                                                                                  #creates corralation table of our DataFrame, calculating correlation values *** allows you to look at companies that are highly correlated, and when they start to deviate invest in one and short the other (or in case of negative correlation when they go in same direction investin one and short the other), or for proper diversification invest in non-correllated stocks
    print(df_corr.head())                                                                                #gets us the inner values of our DataFrame (just columns and rows, no index or header)
    df_corr.to_csv('sp500corr.csv')
    
    data1 = df_corr.values
    fig1 = plt.figure()                                                                                  #specify figure
    ax1 = fig1.add_subplot(111)                                                                          #define axes -- creates 1X1, plot #1

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)                                                     #creates cmap with range from Red(neg) to Yellow(neutral) to Green(pos)
    fig1.colorbar(heatmap1)                                                                              #creates a legend (colorbar that depicts ranges)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)                                         #set xticks at every half mark
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)                                         #set yticks at every half mark
    ax1.invert_yaxis()                                                                                   #removes gap from top of matplotlib graph
    ax1.xaxis.tick_top()                                                                                 #places x axis ticks on top of chart
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)                                                                              #sets color limits for zero correlation (-1) and perfect correlation (1)
    plt.tight_layout()                                                                                   #cleans the graph
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()
#visualize_data()


##Applying Machine Learning to Correlation Tables
def process_data_for_labels(ticker):                                                                     #Prepares Labels
    hm_days = 7   										                                                 #how many days in the future do we have to make or lose 'x%' 
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]   		     #generates % changes

    df.fillna(0, inplace=True)
    return tickers, df
#process_data_for_labels('TSLA')

        
def buy_sell_hold(*args):                                                                                #Helper Function
    cols = [c for c in args]                                                                             #mapping to pandas
    requirement = 0.02                                                                                   #if stock price changes by 2% in 7 days
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):                                                                         #Map Helper Function to DataFrame New Column
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]
                                              ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
    
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)                                                           #replaces any infinite price changes with NAN's
    df. dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()                                            #normalizes price changes (today's value as opposed to yesterday) == % change data for all S&P500 companies including company in question 
    df_vals = df.replace([np.inf, -np.inf], 0)                                                           #replace infinite price changes with 0's
    df_vals.fillna(0, inplace=True)                                                                      #replace NAN's with 0's

    X = df_vals.values                                                                                   #defines feature set  == % change data for all S&P500 companies including company in question
    y = df['{}_target'.format(ticker)].values                                                            #defines target (0, 1 or -1)

    return X, y, df
#extract_featuresets('TSLA')


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

    #clf = neighbors.KNeighborsClassifier()                                                              #uses K means classifier
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())
                            ])
    clf.fit(X_train, y_train)                                                                            #trains data
    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))
          
    return confidence

do_ml('AAPL')   #can run on any ticker -- example here is 'Tesla'
    





    
    
    


    
