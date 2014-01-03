###! /usr/bin/python

import sys
import math
import random
import re
from collections import OrderedDict
import os
import pickle
import time
import nltk
from nltk.stem.porter import PorterStemmer
import itertools
import indexing
import train


if __name__ == "__main__":
        if len(sys.argv)!=2:                # Expect exactly 1 argument: the training data file
                sys.exit(2)
        input1 = file(sys.argv[1],"r")

        indexing.create_index(input1)           ## this function will create dp the basic pre-processing and create the index 
        train.train_classifier()                ## this function will calculate the model parameters for the classifier
