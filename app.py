from os import write
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from arch import arch_model
Nividia = yf.Ticker("NVDA").history(period='5y')
NVDA=Nividia['Close']
Microsoft = yf.Ticker("MSFT").history(period='5y')
MSFT=Microsoft['Close']
Amazon = yf.Ticker("AMZN").history(period='5y')
AMZN=Amazon['Close']
Apple = yf.Ticker("AAPL").history(period='5y')
AAPL=Apple['Close']
Tesla = yf.Ticker("TSLA").history(period='5y')
TSLA=Tesla['Close']
Meta = yf.Ticker("META").history(period='5y')
META=Meta['Close']
Alphabet = yf.Ticker("GOOGL").history(period='5y')
GOOGL=Alphabet['Close'] 

st.title('Quantitative Analysis of Magnificient 7 Stocks')
st.markdown('By Oluwafunso Olalekan')
st.subheader('Introduction')
st.write('The "Magnificent 7" refers to a group of seven leading technology companies NVIDIA (NVDA), Microsoft (MSFT), Amazon (AMZN), Apple (AAPL), Tesla (TSLA), Meta Platforms (META), and Alphabet (GOOGL), known for their significant influence on the broader indices and their role in driving market trends. These seven companies are among the largest in the world by market capitalization and have become key drivers of the overall stock market, particularly within the technology sector. Their performance often reflects broader market trends due to their substantial weighting in major indices like the S&P 500 and Nasdaq.')
st.write('The quantitative analysis of the Magnificent 7 stocks involves the application of mathematical and statistical techniques to evaluate their historical performance, volatility, and potential future trends. I carried out the following analysis using the last 5 year data of the stocks')

data = pd.concat([NVDA,MSFT,AMZN,AAPL,TSLA,META,GOOGL], axis=1)
data.columns = ['NVDA','MSFT','AMZN','AAPL','TSLA','META','GOOGL']
st.subheader('Combined Stock Price Data')
st.write('This shows the closing prices of all seven stocks for the five-year period, gotten from yfinance')
st.dataframe(data)

st.subheader('1. Stock Price comparison')
st.write('The stock price comparison analysis for the Magnificent 7 examines the performance of these leading tech companies over the five-year period. By comparing their stock prices, we can gain insights into their growth trajectories, market volatility, and overall market influence.')
fig, ax = plt.subplots(figsize=(15,7))
NVDA.plot(label='Nividia')
MSFT.plot(label='Microsoft')
AMZN.plot(label='Amazon')
AAPL.plot(label='Apple')
TSLA.plot(label='Tesla')
META.plot(label='Meta')
GOOGL.plot(label='Alphabet')
ax.legend()
st.pyplot(fig)


descriptive_stats = data.describe()
st.subheader('2. Descriptive Statistics of The Magnificient 7 Stocks')
st.write('The descriptive analysis statistics provides a summary of key metrics that describe the distribution and characteristics of the stock prices. The count (number of dataset), mean(i.e average), minimum (the least price the stock traded), standard deviation (which shows how the data deviates from the mean), quartiles (25, 50, & 75), and maximum(the highest price the stock traded for the five year period). ')
st.dataframe(descriptive_stats)
st.write('From the analysis, we can deduce Nvidia lowest trading price (min) as $4.08 and its highest (max) as $135.58.')
st.write('Also, it is important to understand that the standard deviation indicates the extent to which stock prices have deviated from their average; thus, a higher stock price typically corresponds to a greater standard deviation. This is clearly demonstrated by META, which has a standard deviation of 103.47.')


stock_returns = data.pct_change()
Average_daily_return = stock_returns.mean()
fig, ax=plt.subplots(figsize=(15,7))
Average_daily_return.plot(kind='bar', label='Average daily return', color='skyblue' )
st.subheader('3. Average daily return of The magnificient 7 Stocks')
st.write('The average daily return represents the mean return each stock generates on a typical trading day. This metric helps investors understand the potential profitability of each stock on a day-to-day basis. High average daily returns can indicate strong performance, while lower or negative averages may suggest weaker performance or higher risk.')
st.pyplot(fig)
st.write('The analysis reveals that Nvidia leads with the highest average daily return of about 0.32%. Following closely is Tesla, with an average daily return of about 0.29%. In contrast, Amazon has the lowest average daily return at approximately 0.079%.')

import plotly.express as px
Percentage_change = (data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100
Percentage_change_chart=px.bar(Percentage_change, x=Percentage_change.index, y=Percentage_change.values,labels={'x':'Ticker','y':'Percentage change (%)'})
st.subheader('4. Percentage change of The magnificient 7 Stocks')
st.write('The percentage change of the Magnificent 7 stocks refers to the overall change in the stock prices for the last 5 years, expressed as a percentage. This metric helps in assessing how much each stock has gained or lost over that time frame.')
st.plotly_chart(Percentage_change_chart)
st.write('The analysis reveals that Nvidia has achieved an impressive gain of approximately 2,896% over the five-year period. In contrast, Amazon has experienced a significantly lower gain of 99.5% during the same timeframe.')

Covariance=stock_returns.cov()
Standard_deviation=stock_returns.std()
Volatility= Standard_deviation.sort_values(ascending=False)
Volatility_plot=px.bar(Volatility,x=Volatility.index,y=Volatility.values,labels={'x':'Ticker','y':'Volatility'})
st.subheader('5. Volatility of The magnificient 7 Stocks')
st.write('Volatility is a key indicator of risk, as it shows how much a stock price fluctuates over time. The volatility of the Magnificent 7 stocks shows the degree of variation in their stock prices over a specific period, measured by the standard deviation of daily returns.')
st.plotly_chart(Volatility_plot)
st.write('In terms of volatility, measured by the deviation of stocks from their mean, Tesla ranks as the most volatile stock among the seven, with a value of 0.041. Apple and Microsoft seem to exhibit lower volatility, both falling below 0.02.')

Correlation = data.corr()
Correlation_plot=px.imshow(Correlation,labels=dict(x="Ticker", y="Ticker", color="Correlation"),x=Correlation.columns,y=Correlation.columns, color_continuous_scale='RdBu')
st.subheader('6. Correlation of The magnificient 7 Stocks')
st.write('Correlation is a measure that indicates how two stocks move in relation to each other. The correlation of the Magnificent 7 stocks refers to the statistical relationship between the stock price movements of these seven leading tech companies.')
st.plotly_chart(Correlation_plot)
st.write('The correlation analysis shows that Nvidia, Microsoft, Apple, and Alphabet exhibit strong positive correlations, particularly between Microsoft and Apple at 0.93. In contrast, Tesla shows the weakest correlations with the other stocks, indicating that its performance is less influenced by market movements of its peers.')


import plotly.graph_objects as go
Risk_return = pd.DataFrame({'Risk':Standard_deviation, 'Average Daily Return': Average_daily_return })
Risk_return_chart = go.Figure()
Risk_return_chart.add_trace(go.Scatter(
    x=Risk_return['Risk'],
    y=Risk_return['Average Daily Return'],
    mode='markers+text',
    text=Risk_return.index,
    textposition="top center",
    marker=dict(size=10)
))
Risk_return_chart.update_layout(
    title='Risk vs Average Daily Return',
    xaxis_title='Risk',
    yaxis_title='Average Daily Return'
)
st.subheader('7. Risk vs Average Daily Return')
st.write('The Risk vs. Return analysis examines the relationship between the risk associated with each stock and the returns they generate. This analysis helps investors understand the trade-off between the potential rewards of investing in a stock and the level of risk they must assume.')
st.plotly_chart(Risk_return_chart)
st.write('The analysis of risk and average daily return reveals that Nvidia offers the highest average daily return of approximately 0.32% with a risk level of 0.0336, while Tesla carries the highest risk at 0.0414, it provides a significant average daily return of 0.30% but not up to Nvidia. Meanwhile, Microsoft has a lower average daily return of 0.11% and a risk of 0.0191. and Amazon shows the lowest average daily return at 0.08% with a risk of 0.0223. ')


garch_model = {}
for stocks in stock_returns.columns:
  y = stock_returns[stocks].dropna()
  model = arch_model(y, vol='Garch', p=1, q=1, rescale=False)
  model_fit = model.fit()
  garch_model[stocks] = model_fit

forecasts = {}
for stocks, model in garch_model.items():
  forecast = model.forecast(horizon=1)
  forecasts[stocks] = forecast.variance.values[0,0]


n_days = 180
price_forecasts = {}

for stock, model in garch_model.items():
    model=model_fit.model
    sim = model.simulate(model_fit.params, nobs=n_days)
    last_price = data[stock].iloc[-1]

    forecasted_prices = last_price * np.exp(np.cumsum(sim['data']))
    price_forecasts[stock] = forecasted_prices

fig, ax=plt.subplots(figsize=(14, 8))
for stock, prices in price_forecasts.items():
    ax.plot(range(len(prices)), prices, label=stock)
ax.legend(title='Stocks', fontsize=11)
ax.set_xlabel('Days')
ax.set_ylabel('Price')
plt.grid(True, linestyle='--', alpha=0.6)
plt.figtext(0.5, -0.02, "Simulated Stock Prices for the Magnificient 7 Stocks ", ha='center', fontsize=12)
st.subheader('8. Simulated Stock Prices for the Magnificient 7 Stocks (GARCH)')
st.write('The GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model was used to simulate their future stock prices. This simulation captures the potential price movements based on the estimated volatility of each stock. By forecasting these prices over the next six months, the model provides a visual representation of how these seven stocks might behave.')
st.pyplot(fig)
st.write('The analysis forecasts price movements for all seven stocks over the next 180 days, with Meta identified as an outlier, as its price is projected to exceed $800 within the next month. Apple also shows significant potential, with predictions of steady growth over the next four months, eventually soaring beyond $500. In contrast, the other stocks are expected to maintain relatively stable prices, except for Amazon, which is anticipated to decline below its current price within the next six months.')

st.subheader('Conclusion')
st.write('The quantitative analysis of the Magnificent 7 stocks—NVIDIA, Microsoft, Amazon, Apple, Tesla, Meta, and Alphabet—provides valuable insights into their performance, risk, and interrelationships. Together, these analyses offer a comprehensive understanding of the strengths and risks associated with the Magnificent 7, guiding investment strategies in the dynamic tech sector.')
