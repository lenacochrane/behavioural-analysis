
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/digging/digging.xlsx')

df1 = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/analysis/digging/digging.xlsx')

df2 = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/digging/digging.xlsx')


print('N1:')
for condition in df['condition'].unique():
    subset = df[df['condition'] == condition]
    percent = (subset['digging'] == 'Y').mean() * 100
    print(f"{condition} condition percentage: {percent}%")


print('N2:')
for condition in df1['condition'].unique():
    subset = df1[df1['condition'] == condition]
    percent = (subset['digging'] == 'Y').mean() * 100
    print(f"{condition} condition percentage: {percent}%")


print('N3:')
for condition in df2['condition'].unique():
    subset = df2[df2['condition'] == condition]
    percent = (subset['digging'] == 'Y').mean() * 100
    print(f"{condition} condition percentage: {percent}%")




