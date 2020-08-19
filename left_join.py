import pandas as pd
import os

DATA_DIR = 'E:/data/viennacode_combined'

df_1 = pd.read_csv(os.path.join(DATA_DIR, 'combined.csv'))
df_2 = pd.read_csv(os.path.join(DATA_DIR, 'preprocessed_data_index.csv'))

filename_list = df_2['APP_NUM'].tolist()
app_num_list = [int(filename.split('.')[0]) for filename in filename_list]
extention_list = [filename.split('.')[1] for filename in filename_list]
df_2['APP_NUM'] = app_num_list
df_2['EXTENTION'] = extention_list

df = df_2.merge(df_1, on='APP_NUM')
df = df.drop_duplicates(subset='APP_NUM', keep='first')
df.to_csv(os.path.join(DATA_DIR,'merged_index.csv'), index=False, encoding='euc-kr')