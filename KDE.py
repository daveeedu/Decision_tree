import numpy as np
import pandas as pd
from numpy.random import randn
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#1.manually creat KDE by summing the gaussian distribution
ds = randn(100)
sns.rugplot(ds)
plt.hist(ds,alpha=0.5)
plt.savefig('image1.png')

x_axes = np.linspace(ds.min()-1,ds.max()+1,50)

#bandwith creation
bandwith = ((4*ds.std()**5)/(3*len(ds))) ** 0.2

kernels =[]

for point in ds:
    kernel = stats.norm(point,bandwith).pdf(x_axes)
    kernels.append(kernel)

    kernel = kernel/ kernel.max()
    kernel = kernel * 0.6

    plt.plot(x_axes,kernel,alpha=0.5,color="red")


plt.savefig('image6.png')

kde =np.sum(kernels,axis=0)
kde_fig = plt.plot(x_axes,kde,color='green')
sns.rugplot(ds)
plt.suptitle('KDE Plot')
plt.savefig('image7.png')





#2. using seaborn - shortcut
kdefig = sns.kdeplot(ds)
fig = kdefig.get_figure()
fig.savefig('image8.png')
