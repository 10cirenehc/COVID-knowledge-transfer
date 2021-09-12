#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:17:30 2021

@author: ericchen
"""
import pandas as pd

def main(path):
    folder_path = "/Users/ericchen/Desktop/Research/2021-08-23"
    metadata = pd.read_csv(folder_path+ "/metadata.csv")
    metadata = metadata.dropna(subset=['abstract'])
    sample = metadata.head()
    
    corpus = open(path,'a')

    count = 0
    for row in sample.itertuples():
        count = count +1
        if count%1000 ==0:
            print(count)
        abstract_index = 9
        cord_id = 1
        date = 10
        abstract = lemmatize(row[abstract_index])
        corpus.write(abstract)
        corpus.write("\n")
        corpus.write(row[cord_id])
        corpus.write("\n")
        corpus.write(row[date])
        corpus.write("\n")
        corpus.write("\n")

    corpus.close()

def lemmatize(abstract):
    
    import nltk
    from nltk.stem import WordNetLemmatizer
    import string
    lemmatizer = WordNetLemmatizer()
    
    abstract = abstract.translate(str.maketrans(' ', ' ', string.punctuation))
    word_list = nltk.word_tokenize(abstract)
    lemmatized = []
    
    for w in word_list:
        lemmatized.append(lemmatizer.lemmatize(w,get_wordnet_pos(w)))
    
    abstract = ' '.join([w for w in lemmatized])
    return abstract


def get_wordnet_pos(word):
    import nltk
    from nltk.corpus import wordnet

    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

path = "/Users/ericchen/Desktop/Research/to_seg.txt"
main(path)