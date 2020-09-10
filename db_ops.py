from sqlalchemy import create_engine
import pandas as pd

class DataBase:
    def __init__(self, host, user, password, database_name):
        self.host = host
        self.user = user
        self.password = password
        self.database_name = database_name
        self.engine = create_engine(f'mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database_name}')
        self.conn = self.engine.connect()
        self.TABLELIST = []
        self.checkTables()

    def checkTables(self):
        self.TABLELIST = []
        sql = 'SHOW TABLES;'
        result = self.conn.execute(sql)
        row = result.fetchall()
        table_list = []
        for i in row:
            table_list.append(i[0])
        self.TABLELIST = table_list

    def executeSQL(self, sql):
        #create new table: "CREATE TABLE book_details(book_id INT(5), title VARCHAR(20), price INT(5))"
        #alter table(ADD COLUMN): "ALTER TABLE book_details ADD column_name datatype"
        #alter table(MODIFY COLUMN): "ALTER TABLE book_details MODIFY column_name datatype"
        self.conn.execute(sql)

    def appendDataFrameToTable(self, df, table_name):
        df.to_sql(name=table_name, con=self.engine, if_exists='append', index=False)

    def appendDataToTable(self, data, table_name):
        column_names = []
        sql = f'SHOW columns From {table_name};'
        result = self.conn.execute(sql)
        row = result.fetchall()
        for i in row:
            column_names.append(i[0])
        result = ', '.join(column_names)
        sql = f'INSERT INTO {table_name} ({result}) VALUES {data};'
        self.conn.execute(sql)

    def updateTable(self, df, table_name):
        df.to_sql(name=table_name, con=self.engine, if_exists='replace', index=False)

    def exportData(self, table_name):
        return pd.read_sql_table(table_name, self.conn)

    def dropTable(self, table_name):
        sql = f'DROP TABLE {table_name};'
        self.conn.execute(sql)