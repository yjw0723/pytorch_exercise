from sqlalchemy import create_engine
import pandas as pd

class DataBase:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.engine = create_engine(f'mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database}')
        self.conn = self.engine.connect()

    def executeSQL(self, sql):
        #create new table: "CREATE TABLE book_details(book_id INT(5), title VARCHAR(20), price INT(5))"
        #alter table(ADD COLUMN): "ALTER TABLE book_details ADD column_name datatype"
        #alter table(MODIFY COLUMN): "ALTER TABLE book_details MODIFY column_name datatype"
        self.engine.execute(sql)

    def insertDataFrame(self, df, table_name):
        df.to_sql(name=table_name, con=self.engine, if_exists='append', index=False)

    def exportData(self, table_name):
        return pd.read_sql_table(table_name, self.conn)