import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
train_df = pd.read_csv(r"C:\Users\User\Desktop\Metro_Interstate_Traffic_Volume.csv")
print('Dataset shape: ', train_df.shape)
train_df.info()
train_df.head()
train_df.describe(include = 'all')
train_df.isnull().sum()
## Preprocessing of data
from sklearn.preprocessing import LabelEncoder
# drop the unrequired columns
train_df.drop(['date_time', 'weather_description'], axis = 1, inplace = True)
# convert values of day column to numerical format
encoder = LabelEncoder()
train_df['day'] = encoder.fit_transform(train_df['day'])
# subtract 242 from the temp column as there is no temperature below it
train_df['temp'] = train_df['temp'] - 242
# convert the date_time column to datetime type
train_df['date_time'] = pd.to_datetime(train_df['date_time'])
train_df['time'] = train_df['date_time'].dt.hour
fig, (axis1,axis2) = plt.subplots(2, 1, figsize = (20,12))
sns.countplot(x = 'time', data = train_df, ax = axis1, palette="Set3" )
sns.lineplot(x = 'time', y = 'traffic_volume', data = train_df, ax = axis2);

# convert the values of weather_main column to numerical format
encoder = LabelEncoder()
train_df['weather_main'] = encoder.fit_transform(train_df['weather_main'])
