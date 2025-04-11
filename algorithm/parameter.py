


import pandas as pd
label_mean_path = '/home/zty/data/arr_code_0326/train_and_val/mean_std_force_ditribution_train_test.csv'
df = pd.read_csv(label_mean_path, header=None, index_col=None)  
# print(len(df.values),df.values.shape)
label_mean = df.values[0,1:]
label_std = df.values[1,1:]

# print(label_mean)









