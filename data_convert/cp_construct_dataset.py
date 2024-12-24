'''
复制指定txt文件，构成train val test

ptc数据
'''
import os
import numpy as np
import shutil
import glob

source_path = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/ptc_challenge_data'
out_folder = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/raw_data'
scene = 'vesicle' # receptor vesicle
density = 'high'

out_path = os.path.join(out_folder, f'{scene}_{density}')
os.makedirs(out_path, exist_ok=True)

for sub in ['train','val','test']:
    os.makedirs(os.path.join(out_path, sub), exist_ok=True)

# trainval 取自 train_txt; test 取自 challenge_txt
trainvalpath = os.path.join(source_path, 'train_txt')
testpath = os.path.join(source_path, 'challenge_txt')

trainvalfilelist = glob.glob(os.path.join(trainvalpath, f'{scene.upper()}**{density}.txt'))
assert len(trainvalfilelist) == 4, f"{len(trainvalfilelist)},{trainvalfilelist}"
for trainvalfile in trainvalfilelist:
    if '7' in trainvalfile:
        shutil.copy(trainvalfile, os.path.join(out_path,'val',os.path.basename(trainvalfile)))
    else:
        shutil.copy(trainvalfile, os.path.join(out_path,'train',os.path.basename(trainvalfile)))


testfilelist = glob.glob(os.path.join(testpath, f'{scene.upper()}**{density}.txt'))
assert len(testfilelist) == 4
for testfile in testfilelist:
    shutil.copy(testfile, os.path.join(out_path, 'test', os.path.basename(testfile)))


