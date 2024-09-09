from typing import List 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
class CategoricalEncoder:
    def __init__(self,method:str = 'onehot',categories: str = 'auto'):
        self.method = method 
        self.categories = categories 
        self.encoder = {}

    def fit(self,df:pd.DataFrame,columns) -> None:
        for col in columns:
            if self.method == 'onehot':
                self.encoder[col] = OneHotEncoder(sparse_output=False,categories=self.categories)
            elif self.method == 'ordinal':
                self.encoder[col] = OrdinalEncoder(categories=self.categories)
            else:
                raise ValueError('Invalid method. Plese use one of "onehot" or "ordinal".')
            self.encoder[col].fit(df[[col]])

    def transform(self,df,columns):
        df_encoded = df.copy()
        for col in columns:
            transformed = self.encoder[col].transform(df[[col]])
            if self.method == 'onehot':
                transformed = pd.DataFrame(transformed,columns=self.encoder[col].get_feature_names_out([col]))
                df_encoded = pd.concat([df_encoded.drop(columns=[col]),transformed],axis=1)

            else:
                df_encoded[col] = transformed
        return df_encoded 
    
    def fit_transform(self,df,columns):
        self.fit(df,columns)
        return self.transform(df,columns)
    