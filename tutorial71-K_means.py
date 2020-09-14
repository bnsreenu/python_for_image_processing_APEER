# https://youtu.be/H_L7V_BH9pc

"""
@author: Sreenivas Bhattiprolu
"""

import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_excel('data/K_Means.xlsx')
print(df.head())

import seaborn as sns
sns.regplot(x=df['X'], y=df['Y'], fit_reg=False)


from sklearn.cluster import KMeans

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)

model = kmeans.fit(df)

predicted_values = kmeans.predict(df)


plt.scatter(df['X'], df['Y'], c=predicted_values, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', alpha=0.5)
plt.show()


#################################
#Image segmentation using K-means
from skimage import io
import numpy as np
from matplotlib import pyplot as plt

img = io.imread("images/BSE.tif", as_gray=False)
plt.imshow(img, cmap='gray')
# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1, 3))  #-1 reshape means, in this case MxN

#We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
#img2 = np.float32(img2)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
model = kmeans.fit(img2)
predicted_values = kmeans.predict(img2)

#res = center[label.flatten()]
segm_image = predicted_values.reshape((img.shape[0], img.shape[1]))
plt.imshow(segm_image, cmap='gray')
