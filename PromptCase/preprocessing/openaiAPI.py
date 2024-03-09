import tiktoken
import os
from tqdm import tqdm
import openai

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

##########################################################################################################
## insert the openai api key
openai.api_key = " "  
##########################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='2022', help="coliee2022 or coliee2023")
parser.add_argument("--dataset", type=str, default='test', help="train or test")
args = parser.parse_args()


RDIR = "./PromptCase/task1_"+args.dataset+"_"+args.data+"/processed"
WDIR = "./PromptCase/task1_"+args.dataset+"_"+args.data+"/summary_test_"+args.data+"_txt"
files = os.listdir(RDIR)

for pfile in tqdm(files[:]):
    file_name = pfile.split('.')[0]
    if os.path.exists(os.path.join(WDIR, file_name+'.json')):
        # print(pfile, 'already exists')
        pass
    else:
        # print(pfile, 'does not exist')
        with open(os.path.join(RDIR, pfile), 'r') as f:
            long_text = f.read()
            f.close()
        if len(encoding.encode(long_text)) < 500:
            summary_total = long_text
        else:
            summary_total = ''
            length = int(len(encoding.encode(long_text))/3500) + 1
            # Loop through each line in the file
            for i in range(length):
                para = long_text[3500*i:3500*(i+1)]
                for x in range(100):
                    try:
                        completion = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": "Summerize in 50 words:"+para},
                            ]
                        )
                        summary_text = completion.choices[0].message['content']
                        break
                    except:
                        continue    
                summary_total += ' ' + summary_text
        with open(os.path.join(WDIR, file_name+'.txt'), 'w') as file:
            file.write(summary_total)
            file.close()
print('finish')
    