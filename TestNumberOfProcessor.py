# Using the magic encoding
# -*- coding: utf-8 -*-
import copy
import csv
import os
import pickle
import random
import shutil
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import minepy
import numpy as np
import pandas as pd
import sklearn.utils
from deap import creator, base, tools, algorithms
#from deap import creator, base
from scipy import stats
from scoop import futures
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge,
                                  Lasso, RandomizedLasso)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
from timeit import default_timer as timer


print "The number of Processors is:", multiprocessing.cpu_count()
numPro= multiprocessing.cpu_count()
f = open( 'numProcessor.txt', 'w' )
f.write( str(numPro) + '\n' )
f.close()