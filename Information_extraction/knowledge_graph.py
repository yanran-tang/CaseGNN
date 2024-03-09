import sys
import pickle
import os
import glob
from tqdm import tqdm

import en_core_web_trf
# import en_core_web_sm

from lexnlp.extract.en import acts 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='2022', help="coliee2022 or coliee2023")
parser.add_argument("--dataset", type=str, default='test', help="train or test")
parser.add_argument("--feature", type=str, default='issue', help="fact or issue")
args = parser.parse_args()

if args.feature == "fact":
    input_path = './PromptCase/task1_'+args.dataset+'_'+args.data+'/summary_'+args.dataset+'_'+args.data+'_txt'
    output_path = os.getcwd()+'/Information_extraction/coliee'+args.data+'_ie/coliee'+args.data+args.dataset+'_sum'
else:
    input_path = './PromptCase/task1_'+args.dataset+'_'+args.data+'/processed_new'
    output_path = os.getcwd()+'/Information_extraction/coliee'+args.data+'_ie/coliee'+args.data+args.dataset+'_refer'


# print(input_path)
# print(output_path)

class SpacyNER:
    def ner(self,doc):    
        nlp = en_core_web_trf.load()
        doc = nlp(doc)
        return [(X.text, X.label_) for X in doc.ents]
    
    def ner_to_dict(self,ner):
        """
        Expects ner of the form list of tuples 
        """
        ner_dict = {}
        for tup in ner:
            ner_dict[tup[0]] = tup[1]
        return ner_dict
    
    def display(self,ner):
        print(ner)
        print("\n")

def main():
    print("Default ner: spacy")
    
    # os.mkdir("2024train_fact")
    # os.mkdir(output_path)
    # output_path = args.output_path
    ner_pickles_op = output_path + "/ner/"
    kg_doc_op = output_path + "/kg/"
    if os.path.exists(output_path) == False:
        os.mkdir(output_path) 
        os.mkdir(ner_pickles_op)
        os.mkdir(kg_doc_op)

    file_list = []
    for f in glob.glob(input_path+'/*'):
        file_list.append(f)

    for file in tqdm(file_list):
        # if os.path.exists(coref_resolved_op+file.split('/')[-1]):
        #     continue
        # else:
        with open(file,"r") as f:
            lines = f.read().splitlines()
        
        doc = ""
        for line in lines:
            doc += line + ' '
        
        # extract acts
        act_a = acts.get_act_list(doc)
        act_b = []
        if act_a == []:
            act = []
        else:
            act = []
            for tuple in act_a:
                act.append((tuple['act_name'], 'ACT'))
        
        
        spacy_ner = SpacyNER()
        named_entities = spacy_ner.ner(doc)
        named_entities += act
        named_entities = spacy_ner.ner_to_dict(named_entities)
        
        # Save named entities
        op_pickle_filename = ner_pickles_op + "named_entity_" + file.split('/')[-1].split('.')[0] + ".pickle"
        with open(op_pickle_filename,"wb") as f:
            pickle.dump(named_entities, f)
        op_filename = kg_doc_op + file.split('/')[-1]
        with open(op_filename,"w+") as f:
            f.write(doc)

if __name__ == '__main__':
    main()