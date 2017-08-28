import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
df = pd.read_excel("data/data.xlsx", encoding='sys.getfilesystemencoding()')
rcParams['figure.figsize'] = 13, 6
df.sum().plot.barh()
plt.savefig("labels_distribution.png")
plt.show()