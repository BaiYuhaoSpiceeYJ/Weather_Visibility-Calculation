import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50000)

df = pd.read_csv('2020.csv', encoding='utf-8')
df = df.dropna(how='any')
df = df[~df['PM2.5'].str.contains('—')]
df = df[~df['PM10'].str.contains('—')]
df = df[~df['AQI空气质量指数'].str.contains('—')]
# df = df[df['湿度'] > 70]
df = df[['温度', '气压', '湿度', '平均风速', 'PM2.5', 'PM10', 'AQI空气质量指数', '能见度']]
df = pd.DataFrame(df, dtype='float32')
df['C'] = df['AQI空气质量指数'] * df['能见度']

x = np.array(df['C'])
ax = sns.distplot(x)
plt.show()
