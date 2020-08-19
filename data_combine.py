import os
from tqdm import tqdm
import pandas as pd
import database_attribute as DA
from db_ops import DataBase


DATA_DIR = 'E:/data/viennacode'
csv_name_list = os.listdir(DATA_DIR)
db = DataBase(host=DA.host, user=DA.user, password=DA.password, database=DA.database)

# for csv_name in tqdm(csv_name_list[0:20]):
#     path = os.path.join(DATA_DIR, csv_name)
#     df = pd.read_csv(path)
#     df['APP_NUM'] = df["APP_NUM"].astype(str)
#     db.insertDataFrame(df=df, table_name='vienna')

# db = DataBase(host=DA.host, user=DA.user, password=DA.password, database=DA.database)

df = pd.read_sql_table('vienna', db.conn)
save_path = 'E:/data/viennacode_combined/combined.csv'
df.to_csv(save_path, index=False, encoding='euc-kr')