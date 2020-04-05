import requests
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
xs = []
ys = []
def epic():
  
    state_in = input("Please write the full state of where you live. ").strip()
    county_in = input("Please write the full county of where you live. ").strip()
    f = open('covid-19.csv', 'w+')
    print("Retriving data...")
    r = requests.get(URL).content.decode('utf-8')
    print('Data requested.')
    print("Writing data in file...")
    f.write(r)
    print("Written.")
    file = pd.read_csv('covid-19.csv')
    state = file['state']
    county = file['county']
    print("Filtering out data by county and state.")
    for row in file.index:
        if state[row].lower() == state_in.lower() and county_in.lower() == county[row].lower():
            print("Preparing data...")
            ys.append([file['cases'][row], file['deaths'][row]])
    print(ys)
epic()

if len(ys) < 1:
    print("State/county not found. Please type a valid state and county.")
    epic()
for i in range(len(ys)):
    xs.append([i])
print('Data has been prepared to train.')

xs = np.array(xs)
ys = np.array(ys)
xs = np.reshape(xs, (len(xs), 1, 1))

ys = np.reshape(ys, (len(ys), 1, len(ys[0])))
model = Sequential()
model.add(Dense(128, input_shape=(1, 1), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='relu'))
model.compile(loss='mse', optimizer='adam')
model.fit(xs, ys, epochs=5000)
while True:
    cases_deaths = list(ys[-1][0])
    days = input(f"It's been {len(ys)} days since your county has recorded cases and deaths, and there has been {cases_deaths[0]} cases and {cases_deaths[1]} deaths. How far (in days) in the future would you like to simulate and predict? > ")
    try:
        arr = np.array([int(days) + len(ys)]).reshape((1, 1, 1))
        pred = list(model.predict(arr)[0][0])
        print(f"Estimation:{int(pred[0])} cases and {int(pred[1])} deaths")
    except:
        print("Not a valid number. Please input an integer.")
