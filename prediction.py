from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd 

df = pd.read_csv("films_clean2.csv")

# Nettoyage BoxOffice
df['BoxOffice'] = df['BoxOffice'].astype(str).str.replace(' ', '').str.replace(',', '')
df["BoxOffice"] = pd.to_numeric(df["BoxOffice"], errors="coerce")

# Filtrer budget et BoxOffice valides
df = df[(df["budget"] > 0) & (df["BoxOffice"] > 1000000)]

print(f"Films exploitables : {len(df)}")

# Target Encoding acteur et réalisateur
for col in ['Actors_1', 'Director_1']:
    means = df.groupby(col)['BoxOffice'].mean()
    df[col + '_encoded'] = df[col].map(means)

# Encodage du genre
df = pd.get_dummies(df, columns=["Genre_1"])

# Features
genre_cols = [col for col in df.columns if col.startswith("Genre_1_")]
X = df[genre_cols + ["Runtime", "Oscars", "Nominations", "imdbRating", "Released_year", "budget", "Actors_1_encoded", "Director_1_encoded"]]
y = df["BoxOffice"]

X = X.dropna()
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"R² score : {round(score, 3)}")