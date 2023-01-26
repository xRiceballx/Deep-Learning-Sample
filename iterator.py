import pandas as pd
from scipy import signal
from scipy.signal import find_peaks

x = [1,3,2,5,6,7,3,4]

s = iter(x)
        
peaks, _ = find_peaks(s)
n = s[peaks]

i = pd.DataFrame({'value':n})

print(next(n))

