import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_filled = pd.read_csv(r"C:\Users\gobli\Documents\GitHub\AI_lab\project_folder\files\spaceship-titanic\train.csv")
#df_filled = df.copy()
missing_values = df_filled.isnull().sum()
print(missing_values)

num_col = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in num_col:
    mean = df_filled[col].mean()
    df_filled[col] = df_filled[col].fillna(mean)

scaler = MinMaxScaler()
df_filled[num_col] = scaler.fit_transform(df_filled[num_col])

categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for col in categorical_columns:
    mode_val = df_filled[col].mode()[0]
    df_filled[col] = df_filled[col].fillna(mode_val)

true_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
df_encoded = pd.get_dummies(df_filled, columns=true_columns, drop_first=True)

#df_encoded['Deck'] = df_encoded['Cabin'].str.split('/').str[0]
#df_encoded['Side'] = df_encoded['Cabin'].str.split('/').str[2]
#df_encoded = pd.get_dummies(df_encoded, columns=['Deck', 'Side'], drop_first=True)

final_columns_drop = ['Cabin', 'Name', 'PassengerId']
df_final = df_encoded.drop(columns=final_columns_drop, errors='ignore')
df_final.to_csv("processed_titanic_train.csv", index=False)
missing_values = df_final.isnull().sum()
print(missing_values)

input()