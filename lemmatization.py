#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 09:55:49 2021

@author: ericchen
"""
import nltk
from nltk.stem import WordNetLemmatizer
import string
lemmatizer = WordNetLemmatizer()

corpus = open("/Users/ericchen/Desktop/Research/AutoPhrase/data/cord-19-abstracts.txt",'r')
abstracts = []

for x in corpus:
    abstracts.append(x)

export = open("/abstracts_lemmatized.txt", "a")


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
    export.write(abstract)
    export.write("\n")
    
corpus.close()
export.close()