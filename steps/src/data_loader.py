import pandas as pd
from sqlalchemy import create_engine, exc 


class DataLoader:

    def __init__(self,db_uri: str):
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri)
        self.data = None 
    
    def load_data(self,table_name: str) -> pd.DataFrame:
        # load dat from the specific table into df

        query = "SELECT * FROM " + table_name 
        try:
            self.data = pd.read_sql(query, self.engine)
            return self.data 
        except exc.SQLAlchemyErroras as e:
            raise e 
        
    def get_data(self) -> pd.DataFrame:

        if self.data is not None:
            return self.data
        else:
            raise ValueError('No Data loaded yet. Please run load_dat() first.')
