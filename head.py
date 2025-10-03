import pandas as pd

df = pd.read_csv('milk_sessions.csv')

print('### Head ###')
print(df.head())

print('\n### Info ###')
df.info()

print('\n### Describe ###')
print(df.describe())