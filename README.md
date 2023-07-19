# Stock-Market-Predictor
This is a Streamlit app that predicts stock prices using the Facebook Prophet library.

## Description

The stock prediction app allows users to select a stock from a predefined list and predict its future prices. It utilizes historical stock price data obtained from Yahoo Finance using the `yfinance` library. The predictions are made using the Facebook Prophet library.

## Usage

1. Install the necessary dependencies by running the following command: pip install streamlit yfinance prophet plotly
2. Run the app by executing the following command:streamlit run main.py
3. Select a stock from the dropdown menu.
4. Adjust the slider to select the number of years for the prediction.
5. The app displays the historical data, predicted output, forecasted values, and forecast components.

## Requirements

- Python 3.6+
- streamlit
- yfinance
- prophet
- plotly

## Acknowledgments

- This app is built using Streamlit, Facebook Prophet, and Plotly libraries.
- Stock data is obtained from Yahoo Finance.

## Usecases
Prediction of Next Economic Downturn:
Develop a stock market predictor that incorporates a wide range of financial indicators, such as stock prices, market volatility, trading volumes, and macroeconomic factors.
Analyze historical stock market data preceding past economic downturns and identify specific patterns or anomalies that could serve as early warning signs.
Use machine learning algorithms to train the stock market predictor on the historical data and create a predictive model that can anticipate potential economic downturns based on the current market conditions.
Prediction of Early Signs of Collapse of Financial Institution:
Utilize your stock market predictor to monitor and analyze the stock performance of financial institutions, especially those that are publicly traded.
Identify key indicators that may indicate financial distress or potential collapse, such as significant declines in stock prices, high volatility, abnormal trading volumes, or changes in financial ratios.
Develop algorithms or models that continuously evaluate these indicators and generate alerts or warnings when early signs of financial institution collapse are detected.
Prediction of Over Hiring by Companies:
Leverage your stock market predictor to track the stock performance of companies and industries.
Look for correlations between hiring trends and stock performance to identify potential signs of over-hiring by companies.
Combine stock market data with other relevant data sources, such as financial statements, employee turnover rates, and industry-specific indicators, to create a holistic analysis framework that can provide insights into over-hiring practices.
