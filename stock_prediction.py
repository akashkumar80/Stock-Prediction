import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

#Add Company name here
stock_symbol = 'AAPL'

start_date = dt.datetime(2015, 1, 1)
end_date = dt.datetime.now()

data = pdr.data.DataReader(stock_symbol, 'av-daily', start=start_date, end=end_date, api_key='YOUR_ALPHA_VANTAGE_API_KEY')



days=50

data['shift']=data['close'].shift(-days)
data.dropna(inplace=True)
data

X = np.array(data.drop(['shift'],axis=1))
Y = np.array(data['shift'])
X = preprocessing.scale(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)

X = X[:-days]
X_new = X[-days:]

prediction = clf.predict(X_new)

predicted_data=data['close']

final=np.append(predicted_data,prediction)


print(final)
data.dropna(inplace=True)


plt.plot(data['close'],'r',label="original")
plt.plot(final,linewidth=0.5,label='Predicted')
plt.legend(title="Stock Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()