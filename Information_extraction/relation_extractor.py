import os
import subprocess
import glob
import pandas as pd
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='2022', help="coliee2022 or coliee2023")
parser.add_argument("--dataset", type=str, default='test', help="train or test")
parser.add_argument("--feature", type=str, default='issue', help="fact or issue")
args = parser.parse_args()

if args.feature == "fact":
    input_path = os.getcwd()+'/Information_extraction/coliee'+args.data+'_ie/coliee'+args.data+args.dataset+'_sum'
else:
    input_path = os.getcwd()+'/Information_extraction/coliee'+args.data+'_ie/coliee'+args.data+args.dataset+'_refer'


def Stanford_Relation_Extractor():
    
    print('Relation Extraction Started')
    for f in tqdm(glob.glob(input_path+"/kg/*.txt")):
        if os.path.exists(f + '-out.csv'):
            continue
        else:
            os.chdir('./Information_extraction/stanford-openie')

            p = subprocess.Popen(['./process_large_corpus.sh',f,f + '-out.csv'], stdout=subprocess.PIPE)
            output, err = p.communicate()
            
            os.chdir( '../..')
   

    print('Relation Extraction Completed')


if __name__ == '__main__':
    Stanford_Relation_Extractor()
