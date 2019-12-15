import streamlit as st
import numpy as np
import pandas as pd
from glob import glob
from ranker import SimpleRanker, CountRanker, TFRanker, TFIDFRanker, SemanticRanker

results = []

# Functions

@st.cache
def load_dataset(path):
	text = open(path).readlines()
	symbols = "!\"#$%&()*+./:;<=>?@[\]^_`{|}~\n'"
	for ch in symbols:
		text = [x.replace(ch, ' ') for x in text]
	return text

# Sidebar

datasets = sorted(glob('data/*.txt'))
dataset = st.sidebar.selectbox('Choose a dataset', datasets)

text = load_dataset(dataset)

vocab_size = st.sidebar.slider('Set the size of the vocabulary', 1000, 10000, 2000, 1000)
	
ranker_opts = ['Simple Ranker', 'Count Ranker', 'TF Ranker', 'TFIDF Ranker', 'Semantic Ranker']
ranker_opt = st.sidebar.selectbox('Choose a ranker', ranker_opts)

if ranker_opt == 'Simple Ranker':
	ranker = SimpleRanker(text, vocab_size)
elif ranker_opt == 'Count Ranker':
	ranker = CountRanker(text, vocab_size)
elif ranker_opt == 'TF Ranker':
	ranker = TFRanker(text, vocab_size)
elif ranker_opt == 'TFIDF Ranker':
	ranker = TFIDFRanker(text, vocab_size)
elif ranker_opt == 'Semantic Ranker':
	ranker = SemanticRanker(text, vocab_size)
else:
	ranker = None
	
k = st.sidebar.slider('Set the number of results', 1, 100, 10)

# Main Section

st.title('Quick Search')

query = st.text_input('')
if query:
	for row in ranker.search(query, k, return_idx=True):
		st.header('Document: ' + str(row) + ' - Score: ' + str(round(ranker.sim[row],3)))
		st.write(ranker.text[row])
