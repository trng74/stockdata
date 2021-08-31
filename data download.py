# -*- coding: utf-8 -*-
"""
Created on Thu May 13 22:53:08 2021

@author: Trang
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')
#%%
# Import and get ticker list
path = 'C:\\Users\\Trang\\Desktop\\Stock Screen\\Russell3000.csv'
index = pd.read_csv(path)
print(index.head(5))
tickers = index['ticker'].tolist()
#%%
# Download and calculate 1y return
# Create empty dataframe for return
stock_return = pd.DataFrame()

# Iterate over each symbol
for ticker in tickers:
    # Print the current ticker
    print(str(tickers.index(ticker) + 1) + ": " + ticker + " is being downloaded",
          sep=',', end='\n', flush=True)
    
    # Download stock price over 1 year and keep adjusted close price only
    stock = [] # Empty dataframe
    stock = yf.download(ticker, period='1y', interval='3mo', progress=False)
    stock = stock.loc[:, ['Adj Close']]
    stock.rename(columns={'Adj Close':'adj_close'}, inplace=True)
        
    # Calculate 1-year return
    if len(stock) == 0:
        None
    else:
        stock['ticker'] = ticker
        stock['simple_rtn'] = stock['adj_close'][-1]/stock['adj_close'][0] - 1
        stock['log_rtn'] = np.log(stock['adj_close'][-1]/stock['adj_close'][0])
    
    # Add to dataframe
        stock_return = stock_return.append(stock.iloc[-1, :], sort=False)

#%%
# First look at log_return
print(stock_return.shape)

# Print top 10
print(stock_return.sort_values('log_rtn', ascending=False).head(10))

# Print bottom 10
print(stock_return.sort_values('log_rtn', ascending=True).head(10))

# OAS and NAV have unusual returns. I decided to remove them from dataset.
stock_return = stock_return.sort_values('log_rtn', ascending=False)
stock_return = stock_return[2:]
print(stock_return.head(10))

# Print histogram
plt.hist(stock_return['log_rtn'], bins=20)
plt.show()

#%%
# Calculate financial metrics
metrics = pd.DataFrame()
count = 0

# Loop over tickers list, batch of 1000
for ticker in stock_return['ticker']:
    count += 1
    print(str(count) + ": " + ticker + " is being downloaded", sep=',', end='\n', flush=True)
    
    # Download data
    metric = pd.DataFrame()
    ticker_name = ticker
    ticker = yf.Ticker(ticker)
    financial = ticker.financials
    bs = ticker.balance_sheet
    cf = ticker.cashflow
    
    # Assign index to integer
    try:
        rev = financial.index.get_loc('Total Revenue')
        gp = financial.index.get_loc('Gross Profit')
        op = financial.index.get_loc('Operating Income')
        ebit = financial.index.get_loc('Operating Income')
        ni = financial.index.get_loc('Net Income')
        asset = bs.index.get_loc('Total Assets')
        liab = bs.index.get_loc('Total Liab')
        cur_liab = bs.index.get_loc('Total Current Liabilities')
        sltd = bs.index.get_loc('Short Long Term Debt')
        ltd = bs.index.get_loc('Long Term Debt')
        equity = bs.index.get_loc('Total Stockholder Equity')
        depr = cf.index.get_loc('Depreciation')
    except:
        pass

    # Calculate metrics
    if len(financial.iloc[:,1]) == 0:
        None
    else:
        try:
            metric['ticker'] = [ticker_name]
            metric['sector'] = ticker.info['sector']

        # Profitability ratios
            metric['gross_margin'] = [financial.iloc[gp, 1]/financial.iloc[rev, 1]]
            metric['op_margin'] = [financial.iloc[op, 1]/financial.iloc[rev, 1]]
            metric['ni_margin'] = [financial.iloc[ni, 1]/financial.iloc[rev, 1]]
            metric['ebitda_margin'] = [(financial.iloc[ebit, 1] + cf.iloc[depr, 1])
                                       /financial.iloc[rev, 1]]
            metric['roa'] = [financial.iloc[ni, 1]/
                             np.mean((bs.iloc[asset, 1], bs.iloc[asset, 2]))]
            metric['roe'] = [financial.iloc[ni, 1]/
                             np.mean((bs.iloc[equity, 1], bs.iloc[equity, 2]))]
        
        # Debt ratio
            metric['debt_ratio'] = [bs.iloc[liab, 1]/bs.iloc[asset, 1]]
            metric['debt_to_equity'] = [(bs.iloc[sltd, 1] + bs.iloc[ltd, 1])
                                        /bs.iloc[equity, 1]]
    
        # Liquidity ratio
            metric['current_ratio'] = [bs.iloc[cur_liab, 1]/bs.iloc[equity, 1]]
        
        # Growth
            metric['rev_growth'] = [financial.iloc[rev, 1]/financial.iloc[rev, 2] - 1]
            metric['op_growth'] = [financial.iloc[op, 1]/financial.iloc[op, 2] - 1]
            metric['ebitda_growth'] = [(financial.iloc[ebit, 1] + cf.iloc[depr, 1])
                                       /(financial.iloc[ebit, 2] + cf.iloc[depr, 2]) - 1]
            metric['ni_growth'] = [financial.iloc[ni, 1]/financial.iloc[ni, 2] - 1]
            del rev, gp, op, ebit, ni, asset, liab, cur_liab, sltd, ltd, equity, depr
        except:
            pass
        
        # Add to dataframe
        metrics = metrics.append(metric.iloc[-1, :], sort=False)
        
print(metrics.head(10))
#%%

# Export to csv
stock_return.to_csv('C:\\Users\\Trang\\Desktop\\Stock Screen\\Russell3000 1y Return.csv', sep=',')
metrics.to_csv('C:\\Users\\Trang\\Desktop\\Stock Screen\\Russell 3000 Metrics.csv', sep=',')

# Combine to a single df
russell_df = metrics.merge(stock_return, how='left', on='ticker')
russell_df.drop(columns=['adj_close', 'simple_rtn'], inplace=True)

# Export to csv
russell_df.to_csv('C:\\Users\\Trang\\Desktop\\Stock Screen\\Russell 3000 Data.csv', sep=',')

#%%
# Create metrics1 contains LTM metrics
metrics_ltm = pd.DataFrame()
count = 0

# Loop over tickers list
for ticker in df['ticker']:
    count += 1
    print(str(count) + ": " + ticker + " is downloaded", sep=',', end='\n', flush=True)
    
    # Download data
    metric = pd.DataFrame()
    ticker_name = ticker
    ticker = yf.Ticker(ticker)
    financial = ticker.financials
    bs = ticker.balance_sheet
    cf = ticker.cashflow
    
    # Assign index to integer
    try:
        rev = financial.index.get_loc('Total Revenue')
        gp = financial.index.get_loc('Gross Profit')
        op = financial.index.get_loc('Operating Income')
        ebit = financial.index.get_loc('Operating Income')
        ni = financial.index.get_loc('Net Income')
        asset = bs.index.get_loc('Total Assets')
        liab = bs.index.get_loc('Total Liab')
        cur_liab = bs.index.get_loc('Total Current Liabilities')
        sltd = bs.index.get_loc('Short Long Term Debt')
        ltd = bs.index.get_loc('Long Term Debt')
        equity = bs.index.get_loc('Total Stockholder Equity')
        depr = cf.index.get_loc('Depreciation')
    except:
        pass

    # Calculate metrics
    if len(financial.iloc[:,1]) == 0:
        None
    else:
        try:
            metric['ticker'] = [ticker_name]

        # Profitability ratios
            metric['gross_margin_ltm'] = [financial.iloc[gp, 0]/financial.iloc[rev, 0]]
            metric['op_margin_ltm'] = [financial.iloc[op, 0]/financial.iloc[rev, 0]]
            metric['ni_margin_ltm'] = [financial.iloc[ni, 0]/financial.iloc[rev, 0]]
            metric['ebitda_margin_ltm'] = [(financial.iloc[ebit, 0] + cf.iloc[depr, 0])
                                       /financial.iloc[rev, 0]]
            metric['roa_ltm'] = [financial.iloc[ni, 0]/
                             np.mean((bs.iloc[asset, 0], bs.iloc[asset, 1]))]
            metric['roe_ltm'] = [financial.iloc[ni, 0]/
                             np.mean((bs.iloc[equity, 0], bs.iloc[equity, 1]))]
        
        # Debt ratio
            metric['debt_ratio_ltm'] = [bs.iloc[liab, 0]/bs.iloc[asset, 0]]
            metric['debt_to_equity_ltm'] = [(bs.iloc[sltd, 0] + bs.iloc[ltd, 0])
                                        /bs.iloc[equity, 0]]
    
        # Liquidity ratio
            metric['current_ratio_ltm'] = [bs.iloc[cur_liab, 0]/bs.iloc[equity, 0]]
        
        # Growth
            metric['rev_growth_ltm'] = [financial.iloc[rev, 0]/financial.iloc[rev, 1] - 1]
            metric['op_growth_ltm'] = [financial.iloc[op, 0]/financial.iloc[op, 1] - 1]
            metric['ebitda_growth_ltm'] = [(financial.iloc[ebit, 0] + cf.iloc[depr, 0])
                                       /(financial.iloc[ebit, 1] + cf.iloc[depr, 1]) - 1]
            metric['ni_growth_ltm'] = [financial.iloc[ni, 0]/financial.iloc[ni, 1] - 1]
            del rev, gp, op, ebit, ni, asset, liab, cur_liab, sltd, ltd, equity, depr
        except:
            pass
        
        # Add to dataframe
        metrics_ltm = metrics_ltm.append(metric.iloc[-1, :], sort=False)
        
print(metrics_ltm.head(10))

#%%
metrics_ltm.to_csv('C:\\Users\\Trang\\Desktop\\Stock Screen\\Russell 3000 LTM Metrics.csv', sep=',')

#%%
metrics1 = metrics1.set_index('ticker')
preds1 = xg_reg.predict(metrics1)
df1 = metrics1
df1['log_rtn_pred'] = preds1

# Print top 10
print(df1.sort_values('log_rtn_pred', ascending=False).head(10))

# Print bottom 10
print(df1.sort_values('log_rtn_pred', ascending=True).head(10))
