import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Imports and converts the csv into a pandas dataframe
df = pd.read_csv('spotify2010s.csv')

# Since we want unique values, we drop any duplicates, specifying that everything needs to be duplicate
df = df.drop_duplicates()

df['popular_check'] = df['popularity'].apply(lambda x: x > 75)

df_model = df[df['popular_check'] == True].dropna()

# popular_artists = df.groupby(['artists', 'popular_check']).size() \
#     .unstack(fill_value=0) \
#     .sort_values(by=True, ascending=False)
#
# top_pop_artist = popular_artists[True].apply(lambda x: x > 8).to_frame()
#
# top = top_pop_artist.loc[top_pop_artist[True], :]
#
# df_pop_model = df.merge(top, on='artists')
#
# df_pop_model = df_pop_model.drop(columns=['id', True, 'popular_check']).dropna()

X = df_model[['loudness', 'energy', 'acousticness']]
y = df_model[['popularity']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
