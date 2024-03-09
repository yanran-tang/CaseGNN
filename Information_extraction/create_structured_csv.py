from concurrent.futures import process
import pickle
from re import L
from tokenize import Triple
import pandas as pd
import os
import glob
import csv
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

def main():
    if os.path.exists(input_path+"/result/") == False:
        os.mkdir(input_path+"/result/") 
    pickles = []
    for file in glob.glob(input_path+"/ner/*.pickle"):
        pickles.append(file)

    # load each pickle file and create the resultant csv file
    num = 0
    for file in tqdm(pickles):
        with open(file, 'rb') as f:
            entities = pickle.load(f)

        # add all the names in entity set
        entity_set = set(entities.keys())
        file_name_list = file.split('/')[-1].split('.')[0].split('_')[2:]
        file_name = file_name_list[0]
        for word in file_name_list[1:]:
            file_name += '_'
            file_name += word
        
        # parse every row present in the intermediate csv file
        triplet = set()
        with open(input_path+"/kg/" + file_name + ".txt-out.csv", newline='') as csvfile:
            spamreader = csv.reader(csvfile, quotechar='|')
            for row in spamreader:
                if not row:
                    continue
                else:
                    row[0] = row[0].strip()
                    row[2] = row[2].strip()
                    # if relation entity is present in entity set, only then parse futrther
                    if row[0] in entity_set:
                        added = False
                        if type(row[2]) != str:
                            continue
                        else:
                            e2_sentence = row[2].split(' ')
                            # check every word in entity2, and add a new row triplet if it is present in entity2
                            for entity in e2_sentence:
                                if entity in entity_set:
                                    _ = (entities[row[0]], row[0], row[1], entities[entity], row[2])
                                    triplet.add(_)
                                    added = True
                            if not added:
                                _ = (entities[row[0]], row[0], row[1], 'None', row[2])
                                triplet.add(_)
                    elif row[2] in entity_set:
                        added = False
                        if type(row[0]) != str:
                            continue
                        else:
                            e0_sentence = row[0].split(' ')
                            # check every word in entity2, and add a new row triplet if it is present in entity2
                            for entity in e0_sentence:
                                if entity in entity_set:
                                    _ = (entities[entity], row[0], row[1], entities[row[2]], row[2])
                                    triplet.add(_)
                                    added = True
                            if not added:
                                _ = ('None', row[0], row[1], entities[row[2]], row[2])
                                triplet.add(_)
                    else:
                        # check every word in entity2, and add a new row triplet if it is present in entity2
                        _ = ('None', row[0], row[1], 'None', row[2])
                        triplet.add(_)

        processed_pd = pd.DataFrame(list(triplet), columns=['Type', 'Entity1', 'Relationship', 'Type', 'Entity2'])
        processed_pd.to_csv(input_path+'/result/' + file.split("/")[-1].split(".")[0] + '.csv', encoding='utf-8', index=False)

    print("Files processed and saved")

if __name__ == '__main__':
    main()
