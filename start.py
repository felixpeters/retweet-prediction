import os
import numpy as np
import h5py
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
hello = os.getenv("HELLO")
print("Value of environment variable HELLO: {}".format(hello))

data = np.random.random(size=(50, 60))
with h5py.File('/storage/test.hdf5', 'w') as hf:
    hf.create_dataset('test_dataset', data=data)
