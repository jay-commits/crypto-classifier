import pandas as pd
import numpy as np
import ta
data=pd.read_csv('/Users/josephmutui/Desktop/LuxDev/Python-class/crypto-classifier/data/processed/crypto_processed_data.csv')
data.head()
### **1. Price-based Features**
#check 1 day return
data['1_day_return'] = data['close'].pct_change(periods=1)
data[['close', '1_day_return']].head()
#check 7 day return
data['7_day_return'] = data['close'].pct_change(periods=7)
data[['close', '7_day_return']].head(30)
#check rolling volatility 7 
data['7_day_volatility'] = data['1_day_return'].rolling(window=7).std()
data[['1_day_return', '7_day_volatility']].head(15)

### **2. Technical Indicators**

#Using `ta` or `ta-lib`:

# RSI\
 #MACD\
 #Moving averages (SMA20, SMA50, SMA200)\
 #Bollinger Bands\
 #Stochastic Oscillator
data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
data['macd'] = ta.trend.MACD(data['close']).macd()
data['sma20'] = ta.trend.SMAIndicator(data['close'], window=20).sma_indicator()
data['sma50'] = ta.trend.SMAIndicator(data['close'], window=50).sma_indicator()
data['sma200'] = ta.trend.SMAIndicator(data['close'], window=200).sma_indicator()
bb_indicator = ta.volatility.BollingerBands(data['close'])
data['bb_high'] = bb_indicator.bollinger_hband()
data['bb_low'] = bb_indicator.bollinger_lband()
data['stochastic_oscillator'] = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close']).stoch()
data.tail()


#visualize the features of rsi

import matplotlib.pyplot as plt
plt.figure(figsize=(14,7))
plt.subplot(2,1,1)
plt.plot(data['close'], label='Close Price')
plt.title('Close Price with RSI')
plt.legend()
plt.subplot(2,1,2)
plt.plot(data['rsi'], label='RSI', color='orange')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title('Relative Strength Index (RSI)')
plt.legend()
plt.tight_layout()
plt.show()

#plot macd with signal and histogram

plt.figure(figsize=(14,7))
plt.plot(data['macd'], label='MACD', color='blue')
macd_full = ta.trend.MACD(data['close'])
plt.plot(macd_full.macd_signal(), label='Signal Line', color='red')
plt.bar(data.index, macd_full.macd_diff(), label='Histogram', color='gray', alpha=0.5)
plt.title('MACD with Signal Line and Histogram')
plt.legend()
plt.show()

# Visualize the sma20, sma50, sma200 features
plt.figure(figsize=(14,7))
plt.plot(data['close'], label='Close Price', color='blue')  
plt.plot(data['sma20'], label='SMA 20', color='orange')
plt.plot(data['sma50'], label='SMA 50', color='green')
plt.plot(data['sma200'], label='SMA 200', color='red')
plt.title('Simple Moving Averages (SMA)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()


# visualize the bollinger bands
plt.figure(figsize=(14,7))
plt.plot(data['close'], label='Close Price', color='blue')
plt.plot(data['bb_high'], label='Bollinger High Band', color='red', linestyle='--')
plt.plot(data['bb_low'], label='Bollinger Low Band', color='green', linestyle='--')
plt.title('Bollinger Bands')   
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
#visualize the stochastic oscillator with %k and %d
plt.figure(figsize=(14,7))
plt.plot(data['stochastic_oscillator'], label='%K', color='blue') 
stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
plt.plot(stoch.stoch_signal(), label='%D', color='red')
plt.axhline(80, color='red', linestyle='--', alpha=0.8)
plt.axhline(20, color='green', linestyle='--', alpha=0.8)
plt.title('Stochastic Oscillator')
plt.xlabel('Days')
plt.ylabel('Value')
plt.legend()
plt.show()

#drop na values
data= data.dropna().reset_index(drop=True)
data.to_csv('/Users/josephmutui/Desktop/LuxDev/Python-class/crypto-classifier/data/processed/crypto_feature_engineered_data.csv', index=False)