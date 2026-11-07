"""Lei's recording-level family error analysis helpers."""
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'rfi-matplotlib'))
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

MODEL = 'model_matched_clean_fpr'
COUNTS = ('tp', 'fp', 'fn', 'tn')


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def divide(a, b):
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=np.asarray(b) != 0)


def metrics(counts):
    tp, fp, fn, tn = np.asarray(counts).T
    return np.array([divide(tp, tp+fp), divide(tp, tp+fn),
                     divide(2*tp, 2*tp+fp+fn), divide(tp, tp+fp+fn),
                     divide(fp, fp+tn), divide(fn, tp+fn)]).T


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


from IPython.display import display, Image, Markdown
