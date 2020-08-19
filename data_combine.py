import os
import pandas as pd
from tqdm import tqdm
import database_attribute as DA
from db_ops import DataBase

db = DataBase(host=DA.host, user=DA.user, password=DA.password, database=DA.database)

'''INSERT PANDAS DATA FRAME INTO MYSQL TABLE'''
# DATA_DIR = 'E:/data/viennacode'
# csv_name_list = os.listdir(DATA_DIR)
# for csv_name in tqdm(csv_name_list):
#     path = os.path.join(DATA_DIR, csv_name)
#     df = pd.read_csv(path)
#     df['APP_NUM'] = df["APP_NUM"].astype(str)
#     db.insertDataFrame(df=df, table_name='vienna')


'''EXPORT TABLE AND SAVE WITH CSV'''
# df = pd.read_sql_table('vienna', db.conn)
# save_path = 'E:/data/viennacode_combined/combined.csv'
# df.to_csv(save_path, index=False, encoding='euc-kr')
