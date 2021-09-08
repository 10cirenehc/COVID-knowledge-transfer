#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 01:08:50 2021

@author: ericchen
"""
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import re

sns.set_theme()
folder_path = "/Users/ericchen/Desktop/Research/2021-08-23"

metadata = pd.read_csv(folder_path+ "/metadata.csv")
sample= metadata.head()
columns = metadata.columns

plt.figure(figsize=(10, 10), dpi=600)

#%%
#Top 20 Authors
authors = metadata['authors'].dropna()
all_authors = []

for coauthors in authors.tolist():
    current = coauthors.split(";")
    for author in current:
        if bool(re.search(r'\d', author)) == True:
            continue
        if author == "Anonymous,":
            continue
        all_authors.append(author)

top10authors = pd.DataFrame.from_records(
    Counter(all_authors).most_common(20), columns=["Name", "Count"]
)

sns.barplot(x="Count", y="Name", data=top10authors, palette="RdBu_r")
plt.title("Top 20 Authors")

#%%
#Publications Over Time
type(sample['publish_time'][0])
metadata['publish_time_d'] = pd.to_datetime(metadata['publish_time'])

ts1 = metadata[metadata['publish_time_d']> "2000-01-01"]['publish_time_d']
ts1 = ts1.dt.to_period('Y')
ts1 = pd.DataFrame(ts1.value_counts().reset_index())
ts1.columns = ["Year", "Count"]
ts1 = ts1.sort_values('Year')

mask = (metadata['publish_time'].str.len() > 4)
month_df = metadata.loc[mask]

ts2 = month_df[month_df['publish_time_d']> "2019-12-30"]['publish_time_d']
ts2 = ts2.dt.to_period('M')
ts2 = pd.DataFrame(ts2.value_counts().reset_index())
ts2.columns = ["Month", "Count"]
ts2 = ts2.sort_values('Month')

plt.bar(range(len(ts1.Count)), ts1.Count, align='center')
plt.xticks(range(len(ts1.Count)), ts1.Year, size='small', rotation= 'vertical', fontsize = 10)
plt.title("CORD-19 Papers Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Publications")

#%%
#Second time series plot
plt.bar(range(len(ts2.Count)), ts2.Count, align='center')
plt.xticks(range(len(ts2.Count)), ts2.Month, size='small', rotation= 'vertical', fontsize = 10)
plt.title("CORD-19 Papers Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Publications")

#%%
#Top 10 Journals

top10journals = pd.DataFrame.from_records(
    Counter(metadata["journal"]).most_common(20),
    columns=["Journal", "Count"],
)

sns.barplot(x="Count", y="Journal", data=top10journals, palette="RdBu_r")
plt.title("Top 20 Journals")

#%%
#Testing reading the json files
import json

folder_path = "/Users/ericchen/Desktop/Research/06-28"

f= open(folder_path + "/document_parses/pmc_json/PMC35282.xml.json")

data = json.load(f)

#%%
#Network graph

from itertools import combinations
import networkx as nx
import nxviz

authors = authors.tolist()
impact_authors = [None]*len(authors)
author_count =  Counter(all_authors)

for i in range (0,len(authors)):
    impact_authors[i] = authors[i].split(";")
    impact_authors[i] = [author for author in impact_authors[i] 
            if bool(re.search(r'\d', author)) == False and author != "Anonymous," and author_count[author]>3]
    
impact_authors = [i for i in impact_authors if i]

author_connections = list(
    map(lambda x: list(combinations(x[::-1], 2)), impact_authors)
)

flat_connections = [item for sublist in author_connections for item in sublist]

df = pd.DataFrame(flat_connections, columns=["From", "To"])
df_graph = df.groupby(["From", "To"]).size().reset_index()
df_graph.columns = ["From", "To", "Count"]

G = nx.from_pandas_edgelist(
    df_graph, source="From", target="To", edge_attr="Count"
)


# Limit to TOP 70 authors
top70authors = pd.DataFrame.from_records(
    Counter(all_authors).most_common(75), columns=["Name", "Count"]
)

top70_nodes = (n for n in list(G.nodes()) if n in list(top70authors["Name"]))
G_70 = G.subgraph(top70_nodes)

for n in G_70.nodes():
    G_70.nodes[n]["publications"] = int(
        top70authors[top70authors["Name"] == n]["Count"]
    )

c = nxviz.CircosPlot(
    G_70,
    node_grouping="publications",
    edge_width="Count",
    node_color="publications",
)
c.draw()
plt.show()



print("Number of nodes: " + str(G.number_of_nodes()))
print("Number of edges: " + str(G.number_of_edges()))


#%%
nx.write_gexf(G_70, "top70authors.gexf")
nx.write_gexf(G, "author_graph.gexf")

#%%

## Network analysis
deg = nx.degree_centrality(G_70)
bet = nx.betweenness_centrality(G_70)

top_df = pd.DataFrame.from_dict(
    [deg, bet, dict(Counter(all_authors).most_common(70))]
).T
top_df.columns = [
    "Degree Centrality",
    "Betweenness Centrality",
    "Publications",
]

for col in top_df.columns:
    top_df[col] = top_df[col] / max(top_df[col])

top_df = top_df.sort_values("Publications", ascending=False)[:10]
top_df = pd.DataFrame(top_df.stack())
top_df = top_df.reset_index()
top_df.columns = ["Name", "Parameter", "Value"]


fig, ax = plt.subplots(figsize=(10, 8), dpi=600)

sns.barplot(x="Value", y="Name", data=top_df, hue="Parameter", palette="Blues")

plt.title("Top 10 Authors, normalized parameters")
plt.show()

#%%
import math 
count = Counter(all_authors)
author_data = pd.DataFrame.from_dict(count, orient='index').reset_index()
author_data = author_data.rename(columns={'index':'author', 0:'count'})
author_data["cord_uid"] =[[] for _ in range(author_data.shape[0])]
author_data["dates"] = [[] for _ in range(author_data.shape[0])]
author_data["years"] = [[] for _ in range(author_data.shape[0])]

journal_count = Counter(metadata["journal"])
by_journal = metadata.groupby(['journal'])
journal_data = {}

i = 0
journal_data['journals'] = {}
journal_data['id-journal'] = {}

for group_name, df_group in by_journal:
    journal_data['journals'][str(group_name)] = {}
    journal_data['journals'][str(group_name)]['size'] = journal_count[group_name]
    journal_data['journals'][str(group_name)]['publications'] = {}
    journal_data['journals'][str(group_name)]['id'] = i
    journal_data['id-journal'][str(i)] = group_name
    
    for row_index, row in df_group.iterrows():
        journal_data['journals'][str(group_name)]['publications'][str(row['cord_uid'])] = row['publish_time']
    
    i = i+1
                
#%%

#Journal yearly data analysis
import numpy as np
from scipy import stats

metadata['publish_year'] = pd.DatetimeIndex(metadata['publish_time']).year
by_year = metadata.groupby(['publish_year'])
year_counter = Counter(metadata['publish_year'])
date_counter = Counter(metadata['publish_time'])

journal_data['yearly'] = {}
journal_data['yearly']['years'] = []
journal_data['yearly']['number'] = []

for group_name, df_group in by_year:
    count_array = [0]*42856
    df_group = df_group.dropna(subset=['journal'])
    year_count = Counter(df_group['journal'])
    journal_data['yearly']['years'].append(int(group_name))
    
    for journal in year_count:
        count_array[journal_data['journals'][str(journal)]['id']] = year_count[journal]
        
    journal_data['yearly']['number'].append(count_array)
        
journal_data['yearly']['number'] = np.array(journal_data['yearly']['number'])
journal_data['yearly']['percentage'] = journal_data['yearly']['number']/journal_data['yearly']['number'].sum(axis=1)[:,None]
#test = (journal_data['yearly']['percentage'][100].sum())
journal_data['yearly']['entropy'] = [0]*118

for i in range (0,journal_data['yearly']['percentage'].shape[0]):
    journal_data['yearly']['entropy'][i] = stats.entropy(journal_data['yearly']['percentage'][i])
    

plt.plot(journal_data['yearly']['years'][79:116],journal_data['yearly']['entropy'][79:116])
plt.xlabel('Year')
plt.ylabel('Entropy')
plt.title('Entropy of CORD-19 Dataset over years')
plt.xticks(np.arange(1985, 2022, step=2), rotation = 70)
plt.yticks(np.arange(3.5, 10, step=0.5))

print(journal_data['yearly']['percentage'].shape)

#%%
#Plot from 1985
journal_data['total_counts'] = Counter(metadata["journal"])

top_ten = list(journal_data['total_counts'].most_common(6))
del(top_ten[0])

for i in range (0,5):
    journal = top_ten[i][0]
    print(journal)
    index = journal_data['journals'][journal]['id']


    plt.plot(journal_data['yearly']['years'][79:116],journal_data['yearly']['percentage'][79:116,index],label = journal)
    
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Top 5 journals, proportion of dataset over time (1985-2021)')
plt.xticks(np.arange(1985, 2022, step=2), rotation = 70)
plt.xlabel('Year')
plt.yticks(np.arange(0, 0.05, step=0.005))
plt.ylabel('Fraction of dataset')


#%%
#Plot last ten Years
top_ten = list(journal_data['total_counts'].most_common(11))
del(top_ten[0])

for i in range (0,10):
    journal = top_ten[i][0]
    print(journal)
    index = journal_data['journals'][journal]['id']


    plt.plot(journal_data['yearly']['years'][104:116],journal_data['yearly']['percentage'][104:116,index],label = journal)
    
plt.title('Top 10 journals, proportion of dataset over time (2011-2021)')
plt.xticks(np.arange(2011, 2022, step=1), rotation = 70)
plt.xlabel('Year')
plt.yticks(np.arange(0, 0.05, step=0.005))
plt.ylabel('Fraction of dataset')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#%% 

#Setting up the month DF
mask = (metadata['publish_time'].str.len() > 4)
month_df = metadata.loc[mask]

mask = (month_df['publish_year'] == 2020)
df_2020 = month_df.loc[mask]

mask = (month_df['publish_year'] == 2021)
df_2021 = month_df.loc[mask]

df_2020['month'] = pd.DatetimeIndex(df_2020['publish_time']).month
df_2021['month'] = pd.DatetimeIndex(df_2021['publish_time']).month

#%%

#Journal monthly data analysis - 2020


by_month = df_2020.groupby(['month'])

journal_data['2020'] = {}
journal_data['2020']['months'] = []
journal_data['2020']['number'] = []

for group_name, df_group in by_month:
    count_array = [0]*42856
    df_group = df_group.dropna(subset=['journal'])
    month_count = Counter(df_group['journal'])
    journal_data['2020']['months'].append(int(group_name))
    
    for journal in month_count:
        count_array[journal_data['journals'][str(journal)]['id']] = month_count[journal]
        
    journal_data['2020']['number'].append(count_array)
        
journal_data['2020']['number'] = np.array(journal_data['2020']['number'])
journal_data['2020']['percentage'] = journal_data['2020']['number']/journal_data['2020']['number'].sum(axis=1)[:,None]
#test = (journal_data['yearly']['percentage'][100].sum())
journal_data['2020']['entropy'] = [0]*12

for i in range (0,journal_data['2020']['percentage'].shape[0]):
    journal_data['2020']['entropy'][i] = stats.entropy(journal_data['2020']['percentage'][i])


plt.plot(journal_data['2020']['months'],journal_data['2020']['entropy'])
plt.xlabel('Month')
plt.ylabel('Entropy')
plt.title('Entropy of CORD-19 Dataset in 2020')
plt.show()

top_ten = list(journal_data['total_counts'].most_common(11))
del(top_ten[0])

for i in range (0,10):
    journal = top_ten[i][0]
    index = journal_data['journals'][journal]['id']


    plt.plot(journal_data['2020']['months'],journal_data['2020']['percentage'][:,index],label = journal)
    
plt.title('Top 10 journals, proportion of dataset over time (2020 Months)')
plt.xlabel('Month')
plt.ylabel('Fraction of dataset')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#%%

#Journal monthly data analysis - 2021


by_month = df_2021.groupby(['month'])

journal_data['2021'] = {}
journal_data['2021']['months'] = []
journal_data['2021']['number'] = []

for group_name, df_group in by_month:
    count_array = [0]*42856
    df_group = df_group.dropna(subset=['journal'])
    month_count = Counter(df_group['journal'])
    journal_data['2021']['months'].append(int(group_name))
    
    for journal in month_count:
        count_array[journal_data['journals'][str(journal)]['id']] = month_count[journal]
        
    journal_data['2021']['number'].append(count_array)
        
journal_data['2021']['number'] = np.array(journal_data['2021']['number'])
journal_data['2021']['percentage'] = journal_data['2021']['number']/journal_data['2021']['number'].sum(axis=1)[:,None]
#test = (journal_data['yearly']['percentage'][100].sum())
journal_data['2021']['entropy'] = [0]*12

for i in range (0,journal_data['2021']['percentage'].shape[0]):
    journal_data['2021']['entropy'][i] = stats.entropy(journal_data['2021']['percentage'][i])


plt.plot(journal_data['2021']['months'][0:8],journal_data['2021']['entropy'][0:8])
plt.xlabel('Month')
plt.ylabel('Entropy')
plt.title('Entropy of CORD-19 Dataset in 2021')
plt.show()

top_ten = list(journal_data['total_counts'].most_common(11))
del(top_ten[0])

for i in range (0,10):
    journal = top_ten[i][0]
    index = journal_data['journals'][journal]['id']


    plt.plot(journal_data['2021']['months'][0:8],journal_data['2021']['percentage'][0:8,index],label = journal)
    
plt.title('Top 10 journals, proportion of dataset over time (2021 Months)')
plt.xlabel('Month')
plt.ylabel('Fraction of dataset')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#%%

from nltk.stem import WordNetLemmatizer
import nltk
import string
lemmatizer = WordNetLemmatizer()

corpus = open("/Users/ericchen/Desktop/Research/AutoPhrase/data/cord-19-abstracts.txt",'a')

abstracts = metadata['abstract'].dropna()

# Lemmatize with POS Tag
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# 3. Lemmatize a Sentence with the appropriate POS tag
sentence = "review describes the epidemiology and clinical features of 40 patients with cultureproven mycoplasma pneumoniae infections at king abdulaziz university hospital jeddah saudi arabia methods patients with positive m pneumoniae cultures from respiratory specimens from january 1997 through december 1998 were identified through the microbiology"
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
#> ['The', 'strip', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'best']

for item in abstracts.iteritems():
    abstract = item[1].lower()
    abstract = abstract.translate(str.maketrans(' ', ' ', string.punctuation))
    word_list = nltk.word_tokenize(abstract)
    lemmatized = []
    
    for w in word_list:
        lemmatized.append(lemmatizer.lemmatize(w,get_wordnet_pos(w)))
    
    abstract = lemmatized_output = ' '.join([w for w in lemmatized])
    corpus.write(abstract)
    corpus.write("\n")
    
corpus.close()

#%%
corpus = open("/Users/ericchen/Desktop/Research/AutoPhrase/data/cord-19-abstracts_s.txt",'a')

mask = (metadata['publish_year'] == 2018)
df_2019 = metadata.loc[mask]

abstracts = df_2019['abstract'].dropna()

for item in abstracts.iteritems():
    corpus.write(item[1])
    corpus.write("\n")
    
corpus.close()

#%%
log = np.log(np.count_nonzero(journal_data['yearly']['number'], axis=1))
journal_data['yearly']['n_entropy'] = np.asarray(journal_data['yearly']['entropy']).T/log

plt.plot(journal_data['yearly']['years'][79:116],journal_data['yearly']['n_entropy'][79:116])
plt.xlabel('Year')
plt.ylabel('Normalized Entropy')
plt.title('Normalized Entropy of CORD-19 Dataset over years')
plt.xticks(np.arange(1985, 2022, step=2), rotation = 70)

log = np.log(np.count_nonzero(journal_data['2020']['number'], axis=1))
journal_data['2020']['n_entropy'] = np.asarray(journal_data['2020']['entropy']).T/log

plt.plot(journal_data['2020']['months'],journal_data['2020']['n_entropy'])
plt.xlabel('Month')
plt.ylabel('Normalized Entropy')
plt.title('Normalized Entropy of CORD-19 Dataset in 2020')

log = np.log(np.count_nonzero(journal_data['2021']['number'], axis=1))
journal_data['2021']['n_entropy'] = np.asarray(journal_data['2021']['entropy']).T/log

plt.plot(journal_data['2021']['months'][0:8],journal_data['2021']['n_entropy'][0:8])
plt.xlabel('Month')
plt.ylabel('Normalized Entropy')
plt.title('Normalized Entropy of CORD-19 Dataset in 2021')


#