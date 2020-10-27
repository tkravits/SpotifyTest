import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('spotify2010s.csv')

df = df.drop_duplicates()

a = df[df.year.notnull()].groupby('year')['id'].count()

longest_song = df[df.notnull()].groupby('duration_ms')['name'].max()

df_yr_count = df.groupby(['year']).agg(['count'])
#m = df['popularity'] > 75
#df['popular_check'] = np.where(m, 'Y', 'N')

df['popular'] = df['popularity'].apply(lambda x: x > 75)

pop_check_years = df.groupby(['popular', 'year']).size().unstack()
df_pop_check_yr = pop_check_years.loc[True].T

corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
plt.gcf().subplots_adjust(bottom=0.25)
sns.set(font_scale=1)

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
matrix = sns.heatmap(corr, mask=mask, cmap="Spectral", vmax=1, vmin=-1, center=0,
            linewidths=.5, cbar_kws={"shrink": .5}, annot=True, annot_kws = {"fontsize":6})

ax.set_xticklabels(corr.columns, rotation = 45, horizontalalignment='right')

