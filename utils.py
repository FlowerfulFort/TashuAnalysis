from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def encodeFromDataFrame(df, columns):
    encoder = OneHotEncoder(sparse_output=False)
    temp = df.copy()
    encoded = {}
    for c in columns:
        e = encoder.fit_transform(temp[[c]])
        encoded[c] = pd.DataFrame(e, columns=[f'{c}_{i}' for i in range(e.shape[1])])
    return encoded

def Convert2EncodedDF(df, columns):
    encoded = encodeFromDataFrame(df, columns)
    temp = df.drop(columns=columns)
    return pd.concat([temp, *(encoded.values())], axis=1)
