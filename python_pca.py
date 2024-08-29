
import pandas as pd 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

# load features and targets separately
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Data Scaling
x_scaled = StandardScaler().fit_transform(X)

# Reduce from 4 to 3 features with PCA
pca = PCA(n_components=3)

# Fit and transform data
pca_features = pca.fit_transform(x_scaled)

# Bar plot of explained_variance
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )


plt.xlabel('PCA Feature')
plt.ylabel('Explained variance')
plt.title('Feature Explained Variance')
plt.show()


# Scree Plot
# Bar plot of explained_variance
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )

plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red',
    label='Cumulative Explained Variance')

plt.legend(loc='upper left')
plt.xlabel('Number of components')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')

plt.show()


plt.style.use('default')


# Prepare 3D graph
fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot scaled features
xdata = pca_features[:,0]
ydata = pca_features[:,1]
zdata = pca_features[:,2]

# Plot 3D plot
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis')

# Plot title of graph
plt.title(f'3D Scatter of Iris')

# Plot x, y, z even ticks
ticks = np.linspace(-3, 3, num=5)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

# Plot x, y, z labels
ax.set_xlabel('sepal_length', rotation=150)
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_length', rotation=60)
plt.show()


import pandas as pd 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
sns.set()

# load features and targets separately
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Data Scaling
x_scaled = StandardScaler().fit_transform(X)

# Dimention Reduction
pca = PCA(n_components=2)
pca_features = pca.fit_transform(x_scaled)
 
# Show PCA characteristics
print('Shape before PCA: ', x_scaled.shape)
print('Shape after PCA: ', pca_features.shape)
print('PCA Explained variance:', pca.explained_variance_)

# Create PCA DataFrame 
pca_df = pd.DataFrame(
    data=pca_features, 
    columns=[
        'Principal Component 1', 
        'Principal Component 2'
        ])


# Map target names to targets
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}

pca_df['target'] = y
pca_df['target'] = pca_df['target'].map(target_names)
pca_df.sample(10)

# Plot 2D PCA Graph
sns.lmplot(
    x='Principal Component 1', 
    y='Principal Component 2', 
    data=pca_df, 
    hue='target', 
    fit_reg=False, 
    legend=True
    )

plt.title('2D PCA Graph of Iris Dataset')
plt.show()


# Bar plot of explained_variance
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )


plt.xlabel('PCA Feature')
plt.ylabel('Explained variance')
plt.title('Feature Explained Variance')
plt.show()


import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
plt.style.use('default')

# load features and targets separately
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Data Scaling
x_scaled = StandardScaler().fit_transform(X)

# Dimention Reduction
pca = PCA(n_components=3)
pca_features = pca.fit_transform(x_scaled)
 

# Prepare 3D graph
fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot scaled features
xdata = pca_features[:,0]
ydata = pca_features[:,1]
zdata = pca_features[:,2]

# Plot 3D plot
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis')

# Plot title of graph
plt.title(f'3D Scatter of Iris')

# Plot x, y, z even ticks
ticks = np.linspace(-3, 3, num=5)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

# Plot x, y, z labels
ax.set_xlabel('sepal_length', rotation=150)
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_length', rotation=60)
plt.show()



import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sns.set()

# load features and targets separately
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Data Scaling
x_scaled = StandardScaler().fit_transform(X)

# Reduce from 4 to 2 features with PCA
pca = PCA(n_components=2)

# Fit and transform data
pca_features = pca.fit_transform(x_scaled)

# Create dataframe
pca_df = pd.DataFrame(
    data=pca_features, 
    columns=['PC1', 'PC2'])

# map target names to PCA features   
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}

pca_df['target'] = y
pca_df['target'] = pca_df['target'].map(target_names)

# Plot the 2D PCA Scatterplot
sns.lmplot(
    x='PC1', 
    y='PC2', 
    data=pca_df, 
    hue='target', 
    fit_reg=False, 
    legend=True
    )

plt.title('2D PCA Graph')
plt.show()




import numpy as np 
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
plt.style.use('default')
 
# load features and targets separately
iris = datasets.load_iris()
X = iris.data
y = iris.target
 
# Scale Data 
x_scaled = StandardScaler().fit_transform(X)
 
pca = PCA(n_components=3)
 
# Fit and transform data
pca_features = pca.fit_transform(x_scaled)
 
# Create dataframe
pca_df = pd.DataFrame(
    data=pca_features, 
    columns=['PC1', 'PC2', 'PC3'])
 
# map target names to PCA features   
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}
 
# Apply the target names
pca_df['target'] = iris.target
pca_df['target'] = pca_df['target'].map(target_names)
 
# Feature names before PCA
feature_names = iris.feature_names

# Create the scaled PCA dataframe
pca_df_scaled = pca_df.copy()
 
scaler_df = pca_df[['PC1', 'PC2', 'PC3']]
scaler = 1 / (scaler_df.max() - scaler_df.min())
 
for index in scaler.index:
    pca_df_scaled[index] *= scaler[index]
 
# Initialize the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
# Define scaled features as arrays
xdata = pca_df_scaled['PC1']
ydata = pca_df_scaled['PC2']
zdata = pca_df_scaled['PC3']
 
# Plot 3D scatterplot of PCA
ax.scatter3D(
    xdata, 
    ydata, 
    zdata, 
    c=zdata, 
    cmap='Greens', 
    alpha=0.5)
 
# Define the x, y, z variables
loadings = pca.components_
xs = loadings[0]
ys = loadings[1]
zs = loadings[2]
 
# Plot the loadings
for i, varnames in enumerate(feature_names):
    ax.scatter(xs[i], ys[i], zs[i], s=200)
    ax.text(
        xs[i] + 0.1, 
        ys[i] + 0.1, 
        zs[i] + 0.1, 
        varnames)
 
# Plot the arrows
x_arr = np.zeros(len(loadings[0]))
y_arr = z_arr = x_arr
ax.quiver(x_arr, y_arr, z_arr, xs, ys, zs)
 
# Plot title of graph
plt.title(f'3D Biplot of Iris')
 
# Plot x, y, z labels
ax.set_xlabel('Principal component 1', rotation=150)
ax.set_ylabel('Principal component 2')
ax.set_zlabel('Principal component 3', rotation=60)
 
plt.show()



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
  
# Load Iris dataset 
iris = load_iris()
X = iris.data
y = iris.target
  
# Apply PCA with two components 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained Variance Ratio
pca.explained_variance_ratio_
pca.components_

print(abs(pca.components_))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data

# Standardize the features
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Apply PCA and get the explained variance
pca = PCA()
explained_variance = pca.fit(X_std).explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)

# Plot the results
plt.figure(figsize=(10, 6))

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
plt.title('Explained Variance vs. Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(0, 5, 100)

# Define the exponential function (e.g., y = e^(2x))
y = np.exp(2 * x)

import seaborn as sns 
sns.set()
# Plot the exponential curve
plt.plot(x, y)
plt.title('Curse of Dimensionality')
plt.xlabel('Number of Dimensions')
plt.ylabel('Amount of Data Needed')
plt.legend()
plt.xticks([])  # Hide x-axis numbers
plt.yticks([])  # Hide y-axis numbers

plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate two distant data points in 3D space
data = np.array([[0, 0, 0], [100, 100, 100]])

# Apply PCA with only 1 component (reduce to 1D)
pca = PCA(n_components=1)
transformed_data = pca.fit_transform(data)

# Plot the original and transformed points
fig = plt.figure(figsize=(8, 4))

# Plot original 3D points
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', marker='o')
ax1.set_title('Original Data Points')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Z-axis')

# Plot 1D transformed points
ax2 = fig.add_subplot(122)
ax2.scatter(transformed_data, [0, 0], c='red', marker='o')
ax2.set_title('Transformed Data Points (1D PCA)')
ax2.set_xlabel('Principal Component 1')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate data points in 2D space
np.random.seed(42)
data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=100)

# Apply PCA with maximum variance (2 components)
pca_max_var = PCA(n_components=2)
transformed_data_max_var = pca_max_var.fit_transform(data)

# Apply PCA with minimum variance (1 component)
pca_min_var = PCA(n_components=1)
transformed_data_min_var = pca_min_var.fit_transform(data)

# Plot original and transformed points for both cases
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot original 2D points
axes[0, 0].scatter(data[:, 0], data[:, 1], c='blue', marker='o')
axes[0, 0].set_title('Original 2D Data')
axes[0, 0].set_xlabel('X-axis')
axes[0, 0].set_ylabel('Y-axis')

# Plot transformed points with max variance (2D)
axes[0, 1].scatter(transformed_data_max_var[:, 0], transformed_data_max_var[:, 1], c='red', marker='o')
axes[0, 1].set_title('PCA with Max Variance (2 Components)')
axes[0, 1].set_xlabel('Principal Component 1')
axes[0, 1].set_ylabel('Principal Component 2')

# Plot transformed points with min variance (1D)
axes[1, 0].scatter(transformed_data_min_var, np.zeros_like(transformed_data_min_var), c='green', marker='o')
axes[1, 0].set_title('PCA with Min Variance (1 Component)')
axes[1, 0].set_xlabel('Principal Component 1')

# Remove empty subplot
fig.delaxes(axes[1, 1])

plt.tight_layout()
plt.show()
