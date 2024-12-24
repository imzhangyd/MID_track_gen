'''
复制多份config 因为结果的保存是以config的名字保存的
'''

import shutil

source = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/vesicle_low_future1_sample1.yaml'
target = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/configs/vesicle_high_future1_sample1.yaml'

shutil.copy(source, target)