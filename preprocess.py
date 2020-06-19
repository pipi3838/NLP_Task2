import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = '/nfs/nas-5.1/wbcheng/nlp_task2/'
saved_path = '/nfs/nas-5.1/wbcheng/nlp_task2/data/'

data = pd.read_csv(data_dir + 'train.csv', sep=';', engine='python')

# text_len = [len(text) for text in data['Text']]
# print(len(text_len))

# print(max(text_len))  #861
# print(sum(text_len) / len(text_len)) #258
# print(min(text_len)) #52

# print(max(data['Cause_End'])) #594
# print(sum(data['Cause_End']) / len(data['Cause_End'])) #156
# print(max(data['Effect_End'])) #861
# print(sum(data['Effect_End']) / len(data['Effect_End'])) #216

# arranged_data = pd.DataFrame({'id': data['Index'],
#                             'text': data['Text'],
#                             # 'text_len': [len(text) for text in data['Text']],
#                             'cause_start': data['Cause_Start'],
#                             'cause_end' : data['Cause_End'],
#                             'effect_start': data['Effect_Start'],
#                             'effect_end': data['Effect_End']
# })

# train_data, val_data = train_test_split(arranged_data, test_size=0.2)

# train_data.to_csv(saved_path+'train.tsv', sep='\t', index=False)
# val_data.to_csv(saved_path+'val.tsv', sep='\t', index=False)

# data = pd.read_csv(data_dir + 'test.csv', sep=';', engine='python')
# arranged_data = pd.DataFrame({'id': data['Index'], 'text': data['Text']})
# arranged_data.to_csv(saved_path+'test.tsv', sep='\t', index=False)