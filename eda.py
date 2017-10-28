import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from pylab import rcParams

input = "corpus/fb_bank_act_2/data.xlsx"
output = "eda/fb_bank_act_2"
df = pd.read_excel(input, encoding='sys.getfilesystemencoding()')
print("Dataset is loaded!")
rcParams['figure.figsize'] = 13, 6
df.drop("text", axis=1).sum().plot.barh()
plt.savefig(join(output, "labels_distribution.png"))
plt.show()


