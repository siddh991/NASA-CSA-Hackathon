#!/usr/bin/env python
# coding: utf-8

# In[42]:


import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[28]:


class FireWatch(nn.Module):
    def __init__(self):
        super(FireWatch, self).__init__()
        self.input = nn.Linear(10, 35)
        self.hidden1 = nn.Linear(35, 100)
        self.hidden2 = nn.Linear(100, 50)
        self.hidden3 = nn.Linear(50, 25)
        self.output = nn.Linear(25, 3)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        activation1 = self.relu(self.input(x))
        activation2 = self.relu(self.hidden1(activation1))
        activation3 = self.relu(self.hidden2(activation2))
        activation4 = self.relu(self.hidden3(activation3))
        output = self.sigmoid(self.output(activation4))
        return output
        
        


# In[29]:


model = FireWatch()
model


# In[251]:


data = pd.read_csv('./data/newest_data.csv')


# In[252]:


data


# In[1]:


from check_differences import get_differences


# In[5]:


melody = get_differences('2019-08-01-0523_1.tif', '2019-08-02-0736_2.tif')


# In[6]:


melody


# In[79]:


alex = get_differences('index100-2019-05-21-2236.tif', 'index100-2019-05-26-2204.tif')


# In[8]:


alex


# In[9]:


sidd = get_differences('sidd_NDVI_1.tif', 'sidd_NDVI_2.tif')


# In[10]:


sidd


# In[80]:


data


# In[15]:


data.iloc[1085]['result'] = 2


# In[93]:


data.iloc[11]


# In[17]:


data.iloc[1089]['result'] = 2


# In[94]:


x1_index = data[data['windSpeed'] < 7.5]
y[x1_index.index] = 0


# In[95]:


x2_index = data[data['windSpeed'] > 7.5]
y[x2_index.index] = 1


# In[96]:


x3_index = data[data['windSpeed'] > 10]
y[x3_index.index] = 2


# In[99]:


y.value_counts()


# In[59]:


x = data


# In[60]:


x.drop(columns=['Unnamed: 0', 'acq_date', 'acq_time', 'latitude', 'longitude'], inplace=True)


# In[101]:


x


# In[38]:


x.dtypes


# In[73]:


y = pd.Series(np.random.randint(low=1, high=3, size=1090))


# In[75]:


def train(model, x, y, epochs, learning_rate, batch_size, test_size=0.2):
    optim = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    no_of_batches = len(x)/batch_size
    x = torch.from_numpy(x.to_numpy()).type(torch.FloatTensor)
    y = torch.from_numpy(y.to_numpy()).type(torch.LongTensor)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_test, y_train, y_test = x, x, y, y
    for epoch in range(epochs):
        avg_loss = np.array([])
        for batch in range(int(np.floor(no_of_batches))):
            start = batch * batch_size
            end = start + batch_size
            x_ = x_train[start:end]
            y_ = y_train[start:end]
    #       y_ = y_.view(-1)
            hyp = model(x_)
    #         hyp = hyp.view()
            loss = loss_fn(hyp, y_)
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss = np.append(avg_loss, loss.item())
        print(np.average(avg_loss))
    with torch.no_grad():
        correct = 0
        total = 0
        for x_, y_ in zip(x_test, y_test):
            outputs = model(x_)
            print('test:')
            print(y_)
            print(outputs)
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            correct += (predicted == y_).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    


# In[100]:


train(model, x, y, 100, 0.1, 90)


# In[112]:


x


# In[214]:


from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, model_selection

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# x_data = x
# x_data = x_data['windSpeed']
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
train_x, test_x, train_y, test_y = model_selection.train_test_split(x_data,dummy_y,test_size = 0.1, random_state = 0)


# In[228]:


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

input_dim = len(x_data.columns)

model = Sequential()
model.add(Dense(8, input_dim = input_dim , activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_x, train_y, epochs = 50, batch_size = 2)

scores = model.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[229]:


prediction = np.array(x_data.iloc[:100])
predictions = model.predict_classes(prediction)
prediction_ = np.argmax(to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)


# In[230]:


model.predict_classes(prediction)


# In[212]:


x_data.isnull().any()
x_data = x_data.fillna(0)


# In[179]:


from numpy import array
x_new = array(x.loc[11])
x_new = x_new.reshape(10,)
df = x.iloc[[11]]


# In[242]:


import pickle
import joblib
filename = 'sad_model.sav'
joblib.dump(model, filename)


# In[253]:


def predict(index):
    data = pd.read_csv('./data/newest_data.csv')
    data.drop(columns=['Unnamed: 0', 'acq_date', 'acq_time', 'latitude', 'longitude'], inplace=True)
    loaded_model = joblib.load('sad_model.sav')
    this_data = data.loc[[index]]
    result = loaded_model.predict_classes(this_data)
    if result == 0:
        percentage = '0%-120%'
    if result == 1:
        percentage = '120%-250%'
    if result == 2:
        percentage = '250%+'
    return percentage


# In[258]:


per = predict(99)


# In[259]:


per


# In[ ]:




