import sqlite3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from customplot import *

cnx = sqlite3.connect("database.sqlite")
df = pd.read_sql_query("SELECT * FROM player_attributes", cnx)

# display data columns existing
print(df.columns)

# display simple statistics of the dataset
print(df.describe().transpose())

# display how many datapoints are null in each column
print("Null datapoints : \n", df.isnull().sum(axis=0))

# drop null values
# 1.take initial # of rows
rows = df.shape[0]

# drop null values
df = df.dropna()

# check if null values exist
print("Null value exist: ", df.isnull().any().any(),df.shape)

# display how many rows with null values have been deleted
print("Total Rows deleted: ", rows - df.shape[0])

#shuffle rows of df to get distributed sample when we display top rows
df = df.reindex(np.random.permutation(df.index))

#display top 5 rows
print(df.head(5))

#display 2 features(i.e columns) of data for top 10 rows
print(df[:10][["penalties","overall_rating"]])

#display if penalties is correlated to overall rating
print("Penalties-Overall rating Pearson's correlation coefficent:",df['overall_rating'].corr(df['penalties']))

# display correlation of overall rating to few features
potentialFeatures = ['acceleration', 'curve', 'free_kick_accuracy', 'ball_control', 'shot_power', 'stamina']
for f in potentialFeatures:
    related = df['overall_rating'].corr(df[f])
    print("%s : %f" %(f,related))

# plot correlation coeffecient of each feature with overall rating
cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']

correlations = [df['overall_rating'].corr(df[f]) for f in cols]
print(len(cols),len(correlations))

# 2.create a function for plotting a dataframe with string columns and numerical values
def plot_dataframe(df,y_label):
    color = 'coral'
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.ylabel(y_label)

    ax = df2.correlation.plot(linewidth=3.3, color=color)
    ax.set_xticks(df2.index)
    ax.set_xticklabels(df2.attributes, rotation=75);  # Notice the ; (remove it and see what happens !)
    plt.show()

# create a dataframe using cols and correlations
df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations})

# plot above dataframe using the function created
#plot_dataframe(df2,"Overall Rating") TODO commented to avoid blocking next statements

# Define the features you want to use for grouping players
select5features = ['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']
print(select5features)

# Generate a new dataframe by selecting the features you just defined
df_select = df[select5features].copy(deep=True)
print("New dataframe:\n",df_select.head())

# Perform scaling on the dataframe containing the features
data = scale(df_select)
# Define number of clusters
noOfClusters = 4
# Train a model
model = KMeans(init='k-means++', n_clusters=noOfClusters, n_init=20).fit(data)

print(90*'_')
print("\nCount of players in each cluster")
print(90*'_')
print(pd.value_counts(model.labels_, sort=False))

# Create a composite dataframe for plotting
# ... Use custom function declared in customplot.py (which we imported at the beginning of this notebook)
P = pd_centers(featuresUsed=select5features, centers=model.cluster_centers_)
print(P)

# For plotting the graph inside the notebook itself, we use the following command
%matplotlib inline
parallel_plot(P)