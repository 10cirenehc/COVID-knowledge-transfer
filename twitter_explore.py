#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 12:41:14 2021

@author: ericchen
"""
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re
sns.set_theme()



folder_path = "/Users/ericchen/Desktop/Research/Twitter_data/"

columns = ['id','user','time','followers','friends','retweets','favorites','entities', 'sentiment', 'mentions','hashtags','urls']

part_1 = pd.read_csv(folder_path+ "TweetsCOV19.tsv", sep="\t", error_bad_lines=False,header = None, names=columns, index_col=False)
part_2 = pd.read_csv(folder_path+"TweetsCOV19_052020.tsv", sep="\t", error_bad_lines=False, header=None, names=columns, index_col=False)
part_3 = pd.read_csv(folder_path+"TweetsCOV19_3.tsv", sep="\t", error_bad_lines=False, header=None, names=columns, index_col=False)
dfs =[]
dfs.append(part_1)
dfs.append(part_2)
dfs.append(part_3)

data = pd.concat(dfs)
sample = part_1.head()

#%%

#Time series analysis
data['time_d'] = pd.to_datetime(data['time'], format='%a %b %d %H:%M:%S %z %Y')
month = data['time_d']
month = month.dt.to_period('M')
month = pd.DataFrame(month.value_counts().reset_index())
month.head()
month.tail()

month.columns = ["Month", "Count"]
month = month.sort_values('Month')

plt.bar(range(len(month.Count)), month.Count, align='center')
plt.xticks(range(len(month.Count)), month.Month, size='small', rotation= 'vertical', fontsize = 10)
plt.title("Tweets over time")
plt.xlabel("Month")
plt.ylabel("Number of Tweets")

#%%
from rdflib import Graph
g = Graph()
g.parse("http://data.gesis.org/tweetscov19")







    