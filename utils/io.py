import h5py
import json

def save_array(fname, data, dname):
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset(dname, data=data)
    return

def load_array(fname, dname):
    data = []
    with h5py.File(fname, 'r') as hf:
        data = hf[dname][:]
    return data

def save_json(fname, data):
    with open(fname, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return

def load_json(fname):
    data = {}
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

def save_txt(fname, data):
    with open(fname, 'w') as f:
        for s in data:
            f.write(s.encode('unicode_escape').decode())
            f.write("\n")
    return

def load_txt(fname):
    data = []
    with open(fname, 'r') as f:
        for line in f:
            data.append(line)
    return data

