import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input,'yahoo')
clicked = st.button('Enter')



#df=pdr.get_data_alphavantage('AMZN',api_key='IXTFAJJMIX2BM9C8')
#df.to_csv('AMZN.csv')
#df1=df.reset_index()['close']
#plt.plot(df1)
#st.pyplot(df1)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

df1=df.Close

model = load_model('keras_model.h5')


#LSTM are sensitive to the scale of data.So we apply minmax scalar.
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
#converting df1 into an array
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]    
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

#lstm layers
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')




import tensorflow as tf
### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting 
# shift train predictions for plotting


look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back)+1:len(df1)-101, :] = test_predict

plt.plot(scaler.inverse_transform(df1))

scaler=scaler.scale_
scale_factor=1/scaler[0]
train_predict=train_predict*scale_factor
test_predict=test_predict*scale_factor


#train_predict,test_predict=np.array(train_predict),np.array(test_predict)

# plot baseline and predictions
#print(df1)

# st.subheader('Train predict and Test predict plot')
# ma300 = train_predict
# ma400 = test_predict
# fig8 = plt.figure(figsize = (12,6))
# plt.plot(ma300, 'r')
# plt.plot(ma400, 'g')
# plt.plot(df1, 'b')
# st.pyplot(fig8)

#ma300 = df.Close.rolling(100).mean()
#ma400 = df.Close.rolling(100).mean()
#ma500 = df.Close.rolling(100).mean()
#plt.plot(ma300, 'r')
#plt.plot(ma400, 'g')
#plt.plot(ma500, 'b')
#fig8 = plt.figure(figsize = (12,6))

#plt.plot(df.Close, 'b')
#st.pyplot(fig8)

st.subheader('Train predict and Test predict plot')
fig3 = plt.figure(figsize=(12,6))
#plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)


plt.show()
st.pyplot(fig3)
#st.plotly_chart(fig)

x_input=test_data[len(test_data)-100:].reshape(1,-1)
#x_input.shape



temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

day_new=scaler.fit_transform(np.array(day_new).reshape(-1,1))
day_pred=scaler.fit_transform(np.array(day_pred).reshape(-1,1))



scaler=scaler.scale_
scale_factor=1/scaler[0]
day_new=day_new*scale_factor
day_pred=day_pred*scale_factor

# st.subheader('Plot of last 100 days')
# fig4 = plt.figure(figsize=(12,6))
# plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-100:]))
# plt.plot(day_pred,scaler.inverse_transform(lst_output))

st.subheader('Plot of Next 30 days')
fig4 = plt.figure(figsize=(12,6))
plt.plot(day_new)
plt.plot(day_pred)

#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler().fit(lst_output)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
lst_output=scaler.fit_transform(np.array(lst_output).reshape(-1,1))
# st.subheader('Plot of last 100 days')
fig4 = plt.figure(figsize=(12,6))



# scaler.scale_
# scaler=scaler.scale_
# scale_factor=1/scaler[0]
# lst_output=lst_output*scale_factor

#plt.plot(df1[len(df1)-100:])
plt.plot(lst_output)

st.pyplot(fig4)



st.subheader('Final Predicted graph')
df3=df1.tolist()
df3.extend(lst_output)
fig5 = plt.figure(figsize=(12,6))



plt.plot(df3[500:])
st.pyplot(fig5)

# df3=scaler.inverse_transform(df3).tolist()
# fig6 = plt.figure(figsize=(12,6))
# plt.plot(df3)
# st.pyplot(fig6)
