import pandas as pd
import os
from db_ops import *
from database_attribute import user, password, host, database

save_dir = 'E:/data/viennacode_img_19600101_20191231_unique_preprocessed'
img_dir = 'E:/data/viennacode_img_19600101_20191231_unique_preprocessed/imgs'
db = DataBase(user=user, password=password, host=host, database_name=database)
df_1 = db.exportData(table_name='vienna_small_category')
filenames = os.listdir(img_dir)
df_2 = pd.DataFrame({'APP_NUM':filenames})
df_2 = df_2['APP_NUM'].str.split('.',n=1, expand=True)
df_2.columns = ['APP_NUM', 'format']


# DATA_DIR = 'E:/data/viennacode_combined'
#
# df_1 = pd.read_csv(os.path.join(DATA_DIR, 'combined.csv'))
# df_2 = pd.read_csv(os.path.join(DATA_DIR, 'preprocessed_data_index.csv'))
#
# filename_list = df_2['APP_NUM'].tolist()
# app_num_list = [int(filename.split('.')[0]) for filename in filename_list]
# extention_list = [filename.split('.')[1] for filename in filename_list]
# df_2['APP_NUM'] = app_num_list
# df_2['EXTENTION'] = extention_list

df = df_2.merge(df_1, on='APP_NUM')
df = df.drop_duplicates(subset='APP_NUM', keep='first')
df['APP_NUM'] = df['APP_NUM'].astype('str') + '.' + df['format']
df.drop(['format'],axis='columns',inplace=True)
df['VIENNA_CODE'] = df['VIENNA_CODE'].astype('str')
df['APP_NUM'] = df['APP_NUM'].astype('str')
df.to_csv(os.path.join(save_dir,'labels.csv'), index=False, encoding='euc-kr')