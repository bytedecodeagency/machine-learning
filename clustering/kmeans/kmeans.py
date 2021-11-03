import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/data.csv")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

model = KMeans(n_clusters=8, algorithm="full")
model.fit(X_train)

similarities = model.predict(X_test)

X_train, X_test = scaler.inverse_transform(X_train), scaler.inverse_transform(X_test)

print(f"X_train-> {X_train}")
print(f"X_test-> {X_test}")

for i, s in enumerate(similarities):
    print(f'{X_train[s]}->{X_test[i]}')
