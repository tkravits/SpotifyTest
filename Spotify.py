import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Imports and converts the csv into a pandas dataframe
df = pd.read_csv('spotify2010s.csv')

# Since we want unique values, we drop any duplicates, specifying that everything needs to be duplicate
df = df.drop_duplicates()

# Using groupby to calculate the number of songs in each year
list_of_songs_by_yr = df[df.year.notnull()].groupby('year')['id'].count()

# Use groupby to make sure there are no null values for duration and find the longest song by using max()
longest_song = df[df.notnull()].groupby('duration_ms')['name'].max()

#
df_yr_count = df.groupby(['year']).agg(['count'])

df['popular_check'] = df['popularity'].apply(lambda x: x > 75)

pop_check_years = df.groupby(['popular_check', 'year']).size().unstack()
df_pop_check_yr = pop_check_years.loc[True].T

prolific_artists = df.groupby(['artists']).size().sort_values(ascending=False)

table = df.groupby(["artists","popular_check"]).size()\
          .unstack(fill_value=0)\
          .rename_axis(None, axis=1)\
          .reset_index()\
          .sort_values(by=True, ascending=False)
# Use Pandas correlation library to create an initial correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
# plt.gcf().subplots_adjust(bottom=0.25)
# sns.set(font_scale=1)

# Draw the heatmap with the mask and correct aspect ratio
#matrix = sns.heatmap(corr, mask=mask, cmap="Spectral", vmax=1, vmin=-1, center=0,
#            linewidths=.5, cbar_kws={"shrink": .5}, annot=True, annot_kws = {"fontsize":6})

# ax.set_xticklabels(corr.columns, rotation = 45, horizontalalignment='right')

