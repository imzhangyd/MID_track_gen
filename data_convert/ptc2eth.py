'''
将ptc的xml文件转换为eth的txt文件
txt内容
frameid(10的倍数) \t trackid \t posx \t posy
'''

from collections import OrderedDict
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm


def readXML(xml_file_path, t_start=0):
    '''
    读xml得到一个总的df
    '''
    with open(xml_file_path) as file_io:
        lines = file_io.readlines()

    poslist = []
    p = 0
    for i in range(len(lines)):
        if "<particle>" in lines[i]:
            posi = []
        elif "<detection t=" in lines[i]:
            ind1 = lines[i].find('"')
            ind2 = lines[i].find('"', ind1 + 1)
            t = float(lines[i][ind1 + 1 : ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            x = float(lines[i][ind1 + 1 : ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            y = float(lines[i][ind1 + 1 : ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            z = float(lines[i][ind1 + 1 : ind2])
            posi.append([x, y, t, z, float(p)])
        elif "</particle>" in lines[i]:
            p += 1
            poslist.append(posi)
    
    res = pd.DataFrame(columns=['posx', 'posy',' frameid', 'posz', 'trackid'])
    allpoint = []
    for track in poslist:
        for pt in track:
            allpoint.append(pt)
    values = np.array(allpoint)
    res[['posx', 'posy','frameid', 'posz', 'trackid']] = values
    out = res[['frameid','trackid','posx','posy']]
    
    return out


if __name__=='__main__':
    srcpath = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/ptc_challenge_data/challenge_xml'
    trgpath = '/ldap_shared/home/s_zyd/proj_track_gen/MID_track_gen/ptc_challenge_data/challenge_txt'

    allxmlpa = glob.glob(os.path.join(srcpath,'**.xml'))
    for pa in tqdm(allxmlpa):
        res = readXML(pa)
        # print(res)
        res['frameid'] = res['frameid'] * 10
        res = res.sort_values(by='frameid')
        res['frameid'] = res['frameid'].astype('int64')
        trgpa = os.path.join(trgpath, os.path.basename(pa).replace(' ','_').replace('.xml','.txt'))
        res.to_csv(trgpa,sep='\t',header=False,index=False)
