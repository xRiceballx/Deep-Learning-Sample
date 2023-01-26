import numpy as np
import csv
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#reading csv file
with open("./sample2_bcg.csv", "r", encoding="utf_8") as csv_file:
    #リスト形式
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    print(f)