# Source : https://github.com/GuessWhatGame/generic/tree/master/utils

import sys
import pickle
import gzip
import json
import collections


# those file loader/dumper are python2/3 compatibles.

def pickle_dump(data, file_path, gz=False):
    open_fct = open
    if gz:
        open_fct = gzip.open
        file_path += ".gz"

    with open_fct(file_path, "wb") as f:
        pickle.dump(data, f)

def pickle_loader(file_path, gz=False):
    open_fct = open
    if gz:
        open_fct = gzip.open

    with open_fct(file_path, "rb") as f:
        if sys.version_info > (3, 0):  # Workaround to load pickle data python2 -> python3
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            return u.load()
        else:
            return pickle.load(f)

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def dump_json(file_path, json_string):
    with open(file_path, "wb") as f:
        results_json = json.dumps(json_string)
        f.write(results_json.encode('utf8', 'replace'))




