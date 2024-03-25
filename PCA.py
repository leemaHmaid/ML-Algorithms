import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.decomposition import PCA

 

iris = load_iris()
X = iris['data']
y = iris['target']


n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
    )
df["label"] = iris.target

df

df.info()

"""Let's plot our data and see how it's look like"""

column_names = iris.feature_names

plt.figure(figsize=(16,4))
plt.subplot(1, 3, 1)
plt.title(f"{column_names[0]} vs {column_names[1]}")
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.subplot(1, 3, 2)
plt.title(f"{column_names[1]} vs {column_names[2]}")
plt.scatter(X[:, 1], X[:, 3], c=y)
plt.subplot(1, 3, 3)
plt.title(f"{column_names[2]} vs {column_names[3]}")
plt.scatter(X[:, 2], X[:, 3], c=y)
plt.show()

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Display the plot
plt.show()

def mean(X): #  

  # Your code here
  mean = np.sum(X, axis = 0)/X.shape[0]

  return mean

def std(X):  

  # Your code here
  std = (np.sum((X - mean(X)) ** 2, axis = 0)/(X.shape[0]-1))**0.5

  return std

def Standardize_data(X):

  # Your code here
  X_std = (X - mean(X))/std(X)

  return X_std

X_std = Standardize_data(X)

assert (np.round(mean(X_std)) == np.array([0., 0., 0., 0.])).all(), "Your mean computation is incorrect"
assert (np.round(std(X_std)) == np.array([1., 1., 1., 1.])).all(), "Your std computation is incorrect"


def covariance(X):

  ## Your code here
  cov = (1/(X.shape[0]-1))*X.T@X

  return cov

Cov_mat = covariance(X_std)
Cov_mat

 
# Your code here
eigen_values, eigen_vectors = np.linalg.eig(covariance(X_std))  # return eigen values and eigen vectors

print(eigen_values)
print(eigen_vectors)

"""*   rank the eigenvalues and their associated eigenvectors in decreasing order"""

print(eigen_values)
idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]
print(idx)

print("---------------------------------------------------")

eigen_values_sorted = eigen_values[idx]
eigen_vectors_sorted = eigen_vectors.T[:,idx]


explained_variance = [(i / sum(eigen_values))*100 for i in eigen_values_sorted]
explained_variance = np.round(explained_variance, 2)
cum_explained_variance = np.cumsum(explained_variance)

print('Explained variance: {}'.format(explained_variance))
print('Cumulative explained variance: {}'.format(cum_explained_variance))

plt.plot(np.arange(1,X.shape[1]+1), cum_explained_variance, '-o')
plt.xticks(np.arange(1,X.shape[1]+1))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');
plt.grid()
plt.show()

"""#### Project our data onto the subspace"""

# Get our projection matrix
c = 2
P = eigen_vectors_sorted[:c, :] # Projection matrix


X_proj = X_std.dot(P.T)
X_proj.shape

X_proj

plt.title(f"PC1 vs PC2")
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
plt.xlabel('PC1'); plt.xticks([])
plt.ylabel('PC2'); plt.yticks([])
plt.show()



"""## Using sklearn"""

 

#define PCA model to use
pca = PCA(n_components=4)

#fit PCA model to data
pca.fit(X_std)

explained_variance = pca.explained_variance_
print(f"Explained_variance: {explained_variance}")
explained_variance_ratio_percent = pca.explained_variance_ratio_ * 100
print(f"Explained_variance_ratio: {explained_variance_ratio_percent}")
cum_explained_variance = np.cumsum(explained_variance_ratio_percent)

plt.plot(np.arange(1,X_std.shape[1]+1), cum_explained_variance, '-o')
plt.xticks(np.arange(1,X_std.shape[1]+1))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');
plt.grid()
plt.show()

"""*  Kaiser'rule witch keep all the components with eigenvalues greater than 1."""

## Transform data
X_proj = pca.transform(X_std)

plt.title(f"PC1 vs PC2")
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
plt.xlabel('PC1'); plt.xticks([])
plt.ylabel('PC2'); plt.yticks([])
plt.show()



class PCA:

 def __init__(self, n_component):
   self.n_component = n_component
   self.components = None
   self.mean = None

 def fit(self,X):
  self.mean = np.mean(X, axis = 0)
  X = X-  self.mean
  covarianceMatrix = np.cov(X.T)
  eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)
  eigenvectors = eigenvectors.T
  indices = np.argsort(eigenvalues)[::-1]
  self.components = eigenvectors[indices[:self.n_component]]


 def transform(self,X):

    X= X - self.mean
    return np.dot(X, self.components.T)

my_pca = PCA(n_component=2)
my_pca.fit(X)
new_X = my_pca.transform(X)
print(new_X)

