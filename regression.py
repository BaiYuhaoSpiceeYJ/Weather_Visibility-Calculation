from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('2020.csv', encoding='utf-8')
df = df.dropna(how='any')
df = df[~df['PM2.5'].str.contains('—')]
df = df[~df['PM10'].str.contains('—')]
df = df[~df['AQI空气质量指数'].str.contains('—')]
df = df[['温度', '气压', '湿度', '平均风速', 'PM2.5', 'PM10', 'AQI空气质量指数', '能见度']]
df = pd.DataFrame(df, dtype='float32')
df['温度'] = df['温度'] / 30
df['气压'] = df['气压'] / 1000
df['湿度'] = df['湿度'] / 100
df['平均风速'] = df['平均风速'] / 3
df['PM2.5'] = df['PM2.5'] / 100
df['PM10'] = df['PM10'] / 100
df['AQI空气质量指数'] = df['AQI空气质量指数'] / 100
df['能见度'] = df['能见度'] / 10

df = np.array(df, dtype=np.float32)
np.random.shuffle(df)

train_split = df[:int(df.shape[0]*0.8)]
test_split = df[int(df.shape[0]*0.8):]
test_split = test_split[np.argsort(test_split[:, -1])]

x_train = train_split[:, :7]
y_train = train_split[:, 7]
x_test = test_split[:, :7]
y_test = test_split[:, 7]


ploy = PolynomialFeatures(degree=3)
x_train = ploy.fit_transform(x_train)
x_test = ploy.fit_transform(x_test)

liner_model = LinearRegression()

liner_model.fit(x_train, y_train)

y_predict = liner_model.predict(x_test)

sub1 = y_test - y_predict
sub2 = sub1 / y_test

print("mse:", mean_squared_error(y_predict, y_test))

plt.subplot(311)
plt.plot(range(len(y_test)), sorted(y_test), c='k', label='real')
plt.subplot(312)
plt.plot(range(len(y_predict)), y_predict, c='r', label='predict')
plt.subplot(313)
sns.distplot(sub1)
plt.show()

print(np.array(np.abs(sub1)).mean()*10, 'km')
print(np.array(np.abs(sub2)).mean()*100, '%')
