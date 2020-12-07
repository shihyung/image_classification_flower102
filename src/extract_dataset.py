import pandas as pd
import scipy.io
import glob
import os
import tqdm
import shutil

lbs=scipy.io.loadmat('./imagelabels.mat')['labels']
lbs=list(lbs[0])

splits=scipy.io.loadmat('./setid.mat')
train_idx=list(splits['trnid'][0])+list(splits['tstid'][0])
test_idx=list(splits['valid'][0])


flist=glob.glob('./jpgs/*.jpg')
flist.sort()
flist=[f[f.find(os.sep)+1:] for f in flist]

df_all=pd.DataFrame({'image_id':flist,'label':lbs})
df_all.to_csv('dataset_all.csv',index=False)

df_train=pd.DataFrame(columns=['image_id','label'])
df_test=pd.DataFrame(columns=['image_id','label'])
for idx in tqdm.tqdm(range(df_all.shape[0]),total=df_all.shape[0]):
    iid,lb=df_all[['image_id','label']].iloc[idx]
    new_row={'image_id':iid,'label':lb-1} #lb-1 to start from 0
    if (idx+1) in test_idx:
        df_test=df_test.append(new_row,ignore_index=True)
        shutil.move(f'./jpgs/{iid}',f'./test_images/{iid}')
    else:
        df_train=df_train.append(new_row,ignore_index=True)
        shutil.move(f'./jpgs/{iid}',f'./train_images/{iid}')

df_train.to_csv('./train.csv',index=False)
df_test.to_csv('./test.csv',index=False)