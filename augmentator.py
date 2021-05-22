import yaml
import re
import random
import pandas as pd
import numpy as np
import util.tool
import util.data
import util.convert
from svo_extraction.subject_verb_object_extract import findSVOs, get_spacy_nlp_sm_model, invertSentence

class Augmentator():
    def __init__(self, ratio_val, invratio_val, cross_val):
        self.ratio_val = ratio_val
        self.invratio_val = invratio_val
        self.cross_val = cross_val
        self.min_inv_len = 3
        self.dict_list_val = ['./dataset/Panlex/dict/zh2.txt', './dataset/Panlex/dict/ms2.txt']

        idx_dict = util.convert.Common.to_args({"src2tgt": []})
        for dict_file in self.dict_list_val:
            self.get_idx_dict(idx_dict, dict_file, None)

        self.worddict = idx_dict

    def get_idx_dict(self, idx_dict, file, args):
        raw = util.data.Delexicalizer.remove_linefeed(util.data.Reader.read_raw(file))
        idx_dict.src2tgt.append({})
        for line in raw:
            try:
                src, tgt = line.split("\t")
            except:
                src, tgt = line.split(" ")
            
            if src not in idx_dict.src2tgt[-1]:
                idx_dict.src2tgt[-1][src] = [tgt]
            else:
                idx_dict.src2tgt[-1][src].append(tgt)

    def cross(self, x, disable=False):
        if not disable and (self.cross_val >= random.random()):
            lan = random.randint(0,len(self.dict_list_val) - 1)
            if x in self.worddict.src2tgt[lan]:
                return self.worddict.src2tgt[lan][x][random.randint(0,len(self.worddict.src2tgt[lan][x]) - 1)]
            else:
                return x
        else:
            return x

    def invert_str(self, x, disable=False):
        if len(x) < self.min_inv_len:
            return x
        else:
            x = " ".join(x)
            return invertSentence(x).split()

    def cross_list(self, x):
        x = x.split()
        length = len(x)
        utter = self.invert_str(x, not (self.invratio_val >= random.random()))
        if len(utter) == length:
            x = utter
        else:
            x = x
        return " ".join([self.cross(xx, not (self.ratio_val >= random.random())) for xx in x])


with open("./rasa/data/in_nlu.yml") as read_file:
    augmentator = Augmentator(0.75, 0.75, 0.75)
    documents = yaml.full_load(read_file)
    df = pd.DataFrame(np.empty((0,2)), columns=["intent", "example"])
    for entry in documents.get('nlu'):
        intent = entry.get('intent')
        examples = entry.get('examples')
        examples = re.split('\n- |- |\n',examples)[1:-1]
        examples_list = [augmentator.cross_list(example) for example in examples]
        # examples_list = [example for example in examples]
        df = df.append(pd.concat([pd.DataFrame([[intent, example]], columns=("intent", "example")) for example in examples_list]), ignore_index=True)

    with open ("./rasa/data/nlu.yml", "w") as outputFile:
        outputFile.write('version: "2.0"\n\nnlu:\n')
        for intent in df["intent"].unique():
            outputFile.write("\n- intent: %s\n  examples: |\n"%intent)
            for sentence in df[df["intent"]==intent]["example"]:
                outputFile.write("    - %s\n"%sentence)


