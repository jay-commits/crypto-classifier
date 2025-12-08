# **Step 4 --- Label Generation (Target Variable)**
import pandas as pd

data = pd.read_csv('/Users/josephmutui/Desktop/LuxDev/Python-class/crypto-classifier/data/processed/crypto_feature_engineered_data.csv')

data["future_return"] = data["close"].pct_change().shift(-1)

def label(row):
  if row["future_return"] > 0.04:
        return 2
  elif row["future_return"] < -0.06:
        return 0
  else:
        return 1

data["label"] = data.apply(label, axis=1)
data['label_name'] = data['label'].map({0:'SELL', 1:'HOLD', 2:'BUY'})
data = data.dropna(subset=['future_return'])
print(data.head())
data.to_csv('/Users/josephmutui/Desktop/LuxDev/Python-class/crypto-classifier/data/processed/crypto_labeled_data.csv', index=False)

  
    #if row["future_return"] > 0.02:
        #return 2
    #elif row["future_return"] < -0.02:
        #return 0
    #else:
        #return 1