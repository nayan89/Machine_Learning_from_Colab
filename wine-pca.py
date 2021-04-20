import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
redwine = pd.read_csv("Data/winequality-red.csv", delimiter=";")
print(redwine.head())

X_red = redwine.iloc[:,0:11]
y_red = redwine.iloc[:,-1]
#%%
# function to evaluate pca
def PCA(X, n_components):
    X_mean = X-np.mean(X, axis=0)
    cov = np.cov(X_mean, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    sorted_idx = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_idx]
    sorted_eigenvector = eigen_vectors[:,sorted_idx]
    eigenvector_subset = sorted_eigenvector[:,0:n_components]
    X_reduced = np.dot(eigenvector_subset.transpose(), X_mean.transpose()).transpose()
    return X_reduced

# pca for red wine dataset
pca = PCA(X_red, 2)
principal_component = pd.DataFrame(pca, columns=['PC1','PC2'])
print(principal_component.head())
#%%
plt.figure(figsize=(12,6))
plt.scatter(principal_component['PC1'], principal_component['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('red wine principal component scatter plot')
plt.show()
#%%
# pca for white wine datasets
whitewine = pd.read_csv("Data/winequality-white.csv", delimiter=';')
print(whitewine.head())
#%%
X_white = whitewine.iloc[:,0:11]
y_white = whitewine.iloc[:,-1]
#%%
pca_white = PCA(X_white, 2)
principal_component_w = pd.DataFrame(pca_white, columns=['PC1','PC2'])
print(principal_component_w.head())
#%%
plt.figure(figsize=(12,6))
plt.scatter(principal_component_w['PC1'], principal_component_w['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('white wine principal components scatter plot')
plt.show()
#%%
# varianace explained for white wine data
X_mean = X_white-np.mean(X_white, axis=0)
cov = np.cov(X_mean, rowvar=False)
eigen_values, eigen_vectors = np.linalg.eig(cov)
total_eigenval = sum(eigen_values)
exp_var = [(i/total_eigenval) for i in sorted(eigen_values, reverse=True)]
print("white wine variance explained")
print(exp_var)
#%%
# variance explained for red wine data
X_mean = X_red-np.mean(X_red, axis=0)
cov = np.cov(X_mean, rowvar=False)
eigen_values, eigen_vectors = np.linalg.eig(cov)
total_eigenval = sum(eigen_values)
exp_var = [(i/total_eigenval) for i in sorted(eigen_values, reverse=True)]
print("red wine variance explained")
print(exp_var)