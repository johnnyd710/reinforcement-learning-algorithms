import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data0 = pd.read_csv('data_b0.csv', sep=',', header=None)
data1 = pd.read_csv('data_b1.csv', sep=',', header=None)
data2 = pd.read_csv('data_b2.csv', sep=',', header=None)
data3 = pd.read_csv('data_b3.csv', sep=',', header=None)
data = pd.DataFrame(index = range(500))
data['32'] = data0
data['15'] = data1
data['5'] = data2
data['1'] = data3

data.plot()
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.show()
