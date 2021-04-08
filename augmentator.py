import yaml
import re
import random
import util.tool
import util.data
import util.convert
from svo_extraction.subject_verb_object_extract import findSVOs, get_spacy_nlp_sm_model, invertSentence

class Augmentator():
    def __init__(self):
        self.ratio_val = 1.0
        self.invratio_val = 1.0
        self.cross_val = 1.0
        self.min_inv_len = 3
        self.dict_list_val = ['./dataset/Panlex/dict/th2.txt', './dataset/Panlex/dict/es2.txt']

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
            return invertSentence(x).split(" ")

    def cross_list(self, x):
        length = len(x)
        utter = self.invert_str(x, not (self.invratio_val >= random.random()))
        if len(utter) == length:
            x = utter
        else:
            x = x
        return [self.cross(xx, not (self.ratio_val >= random.random())) for xx in x]


with open("./rasa/data/in_nlu.yml") as read_file:
    augmentator = Augmentator()
    documents = yaml.full_load(read_file)
    # print (documents)
    nlu = []
    for entry in documents.get('nlu'):
        intent = entry.get('intent')
        examples = entry.get('examples')
        examples = re.split('\n- |- |\n',examples)[1:-1]
        examples_list = augmentator.cross_list(examples)
        examples_list = "- " + "\n- ".join(examples_list) + "\n"
        nlu.append({'intent': intent, 'examples': examples_list})
    output_dict = {'version': '2.0', 'nlu': nlu}

    with open("./rasa/data/nlu.yml", "w") as write_file:
        yaml.safe_dump(output_dict, write_file, default_style=None, default_flow_style=False)



