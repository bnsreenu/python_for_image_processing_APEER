#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

####################################
#
#For better control over plotting you may as well use Matplotlib or Seaborn
# pip install seaborn

##########################################
#Seaborn builds on top of matplotlib to provide a richer out of the box environment. 
# https://seaborn.pydata.org/
#https://seaborn.pydata.org/examples/index.html   #Checkout for more examples


import pandas as pd
df = pd.read_csv('data/manual_vs_auto.csv')

import seaborn as sns
from matplotlib import pyplot as plt
##############
#Single variable (distribution histogram plots)
#sns.distplot(df['Manual'])  #Will fail as we have a few missing values.

print(df.isnull()) #To find out if we have any null values in each column

df.isnull().values.any() #To find out if we have any null values

df.isnull().sum() #Most useful. Tells us where we have null values.

df = df.drop(['Manual2'], axis=1)
#Let us fill missing values with a value of 100
df['Manual'].fillna(100, inplace=True)
###########################################################
#Distribution plot (Histogram)
sns.distplot(df['Manual'])   #The overlay over histogram is KDE plot (Kernel density distribution)

#Making it visually appealing
sns.distplot(df['Manual'], bins=20, kde=True, rug=False, hist_kws=dict(edgecolor='k', linewidth=0.8)) 

plt.xlim([80,120])
sns.distplot(df['Manual'], bins=20, kde=True, rug=False, hist_kws=dict(edgecolor='k', linewidth=0.8)) 

################################################################
#KDE plots. Kernel density estimation.
#KDE is a way to estimate the probability density function of a continuous random variable.

import pandas as pd
df = pd.read_csv('data/manual_vs_auto.csv')
df['Manual'].fillna(100, inplace=True)

import seaborn as sns
sns.kdeplot(df['Manual'], shade=True)

## Add Multiple plots
sns.kdeplot(df['Auto_th_2'], shade=True)
sns.kdeplot(df['Auto_th_3'], shade=True)
sns.kdeplot(df['Auto_th_4'], shade=True)
##########################################################################

#Basic line plot
import pandas as pd
df = pd.read_csv('data/manual_vs_auto.csv')
df['Manual'].fillna(100, inplace=True)

import seaborn as sns
sns.set(style='darkgrid')   #Adds a grid
sns.lineplot(x='Image', y='Manual', data=df, hue='Unnamed: 0')   #Simple line plot
#Hue tells seaborn how to color various subcategories, like our set in this example.


####################################################################            
#Joint plots - Scatter plots showing Gaussian distribution of sample space.

import pandas as pd
df = pd.read_csv('data/manual_vs_auto.csv')
df['Manual'].fillna(100, inplace=True)
import seaborn as sns

#Basic scatter plot with density curve.
sns.jointplot(x="Manual", y="Auto_th_2", data=df, kind='reg', color='r')

#KDE plot, Kernel density estimation.
sns.jointplot(x="Manual", y="Auto_th_2", data=df, kind="kde")

################################################################################

#Scatterplot with linear regression

import pandas as pd
df = pd.read_csv('data/manual_vs_auto.csv')
df['Manual'].fillna(100, inplace=True)

#Change Unnamed: 0 name to Image_set
df = df.rename(columns = {'Unnamed: 0':'Image_set'})
import seaborn as sns

#Scatter Plot with linear regression fit. Change order for higher order fits.
sns.lmplot(x='Manual', y='Auto_th_2', data=df, order=1)

#Scatterplot with linear regression fit 
#Separated by hue (e.g. Image_set)
# 95% confidence interval for each set
sns.lmplot(x='Manual', y='Auto_th_2', data=df, hue='Image_set', order=1)  

#If you want equation, not possible to display in seaborn but you can get it the
#regular way using scipy stats module. 
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Manual'],df['Auto_th_2'])
print(slope, intercept)

#Regplots are similar to lmplots. 
sns.regplot(x='Manual', y='Auto_th_2', data=df, color='g')

###############################################################
#Relationship between each feature and another selected feature can be easily plotted
#using pariplot function in Seaborn

import pandas as pd
import seaborn as sns

df = pd.read_csv('data/manual_vs_auto.csv')
df = df.drop(['Manual2'], axis=1)
df['Manual'].fillna(100, inplace=True)
print(df.columns)


sns.pairplot(df, x_vars=["Auto_th_2", "Auto_th_3", "Auto_th_4"], y_vars="Manual")

#too small. Let us chage the size

sns.pairplot(df, x_vars=["Auto_th_2", "Auto_th_3", "Auto_th_4"], y_vars="Manual", size=6, aspect=0.75)

#Change Unnamed: 0 name to Image_set
df = df.rename(columns = {'Unnamed: 0':'Image_set'})

#Generate a grid with liner relationship between each column (feature)
sns.pairplot(df, hue='Image_set', dropna=True)

#######################################################################
#Swarm plots
#Let's use manual_vs_auto2 file that we generated earlier 
import pandas as pd
df = pd.read_csv('data/manual_vs_auto2.csv')
df['Manual'].fillna(100, inplace=True)
print(df.head())

import seaborn as sns

sns.swarmplot(x = "Image_set", y="Manual", data = df, hue="cell_count_index")

#SPlit each category
sns.swarmplot(x = "Image_set", y="Manual", data = df, hue="cell_count_index", dodge=True)


###########################################################
"""
we can utilise the pandas Corr() to find the correlation between each variable 
in the matrix and plot this using Seaborn’s Heatmap function, 
specifying the labels and the Heatmap colour range.

"""

import pandas as pd
df = pd.read_csv('data/manual_vs_auto.csv')
print(df.dtypes)
df['Manual'].fillna(100, inplace=True)
#Change Unnamed: 0 name to Image_set
df = df.rename(columns = {'Unnamed: 0':'Image_set'})

import seaborn as sns
corr = df.loc[:, df.dtypes == 'int64'].corr() #Correlates all int64 columns

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
##########################
















