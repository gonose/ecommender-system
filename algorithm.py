# General import and load data
import pandas as pd
import numpy as np
import os
import io

from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise import dump

from surprise import KNNBaseline

(predictions, algorithm) = dump.load("algorithm")

print(type(algorithm))

def predict(uid, iid):
	print(type(algorithm))
	predicted = algorithm.predict(uid, iid, verbose=True)
	return predicted
