##############		library installation		##############
##############################################################

# conda install -c anaconda joblib networkx
# conda install -c conda-forge nltk gensim papermill  python-louvain
# pip install --upgrade gensim
# pip install jupyter


# import nltk
# nltk.download('stopwords')

##############		Imports		##############
##############################################


import os, sys, getopt, copy, pickle, gc, errno, shutil
import functools
import pandas as pd
import numpy as np

from datetime import datetime, date, timezone

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Lock
from pandarallel import pandarallel

import signal
import requests
import itertools
import operator
import tempfile
import re

from pprint import pprint
from IPython.display import display, HTML
import matplotlib.pyplot as plt



from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


from sklearn.feature_extraction.text import CountVectorizer



import networkx as nx
import community

import dask.dataframe as dd
import errno

# nlp
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess