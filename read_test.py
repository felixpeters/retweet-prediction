import os
import numpy as np
import h5py
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

with h5py.File('/storage/test.hdf5', 'r') as hf:
    data = hf['test_dataset'][:]
    print(type(data))
    print(data.shape)
