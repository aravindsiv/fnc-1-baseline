

"""
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True)
model.wv.similarity('woman', 'man')


def wordtovec(hd_body_dict,max_len):
    non_words=set()
    for k in hd_body_dict["hd_tok"]:
        if(k.lower() in model.vocab ):
            for m in hd_body_dict["body_tok"]:
                if(m.lower() in model.vocab)
                sim_set.add()
"""

def common_noun_feature(hd_body_dict,max_len):
    hd_noun=set()
    body_noun=set()
    for i,k in enumerate(hd_body_dict["body_pos"]):
        if(k in ["PROPN","NOUN"]):
            if(hd_body_dict["body_tok"][i] in ["reports","report","reported","reportedly"] ):
                continue
            body_noun.add(hd_body_dict["body_tok"][i].lower())
    for i,k in enumerate(hd_body_dict["hd_pos"]):
        if(k in ["PROPN","NOUN"]):
            if(hd_body_dict["hd_tok"][i] in ["reports","report","reported","reportedly"] ):
                continue
            hd_noun.add(hd_body_dict["hd_tok"][i].lower())
    #return(len(hd_noun.intersection(body_noun))/len(hd_body_dict["hd_pos"]))
    if(len(hd_noun)==0):
        return 0
    return(len(hd_noun.intersection(body_noun))/len(hd_noun))
    #return (len(hd_noun.intersection(body_noun)) / max_len)

def common_verb_feature(hd_body_dict,max_len):
    hd_verb=set()
    body_verb=set()
    for i,k in enumerate(hd_body_dict["body_pos"]):
        if(k in ["ADV","VERB"]):
            if(hd_body_dict["body_tok"][i] in ["reports","report","reported","reportedly"] ):
                continue
            body_verb.add(hd_body_dict["body_tok"][i].lower())

    for i,k in enumerate(hd_body_dict["hd_pos"]):
        if(k in ["ADV","VERB"]):
            if(hd_body_dict["hd_tok"][i] in ["reports","report","reported","reportedly"] ):
                continue
            hd_verb.add(hd_body_dict["hd_tok"][i].lower())
    #return(len(hd_verb.intersection(body_verb))/len(hd_body_dict["hd_pos"]))
    #return (len(hd_verb.intersection(body_verb)) / max_len)
    if(len(hd_verb)==0):
        return 0
    return(len(hd_verb.intersection(body_verb))/len(hd_verb))

def common_adjective_feature(hd_body_dict,max_len):
    hd_adj=set()
    body_adj=set()
    print(hd_body_dict["hd_tok"])
    for i,k in enumerate(hd_body_dict["body_pos"]):
        if(k in ["ADJ"]):
            body_adj.add(hd_body_dict["body_tok"][i].lower())
    for i,k in enumerate(hd_body_dict["hd_pos"]):
        if(k in ["ADJ"]):
            hd_adj.add(hd_body_dict["hd_tok"][i].lower())
    #return(len(hd_adj.intersection(body_adj))/len(hd_body_dict["hd_pos"]))
    #return (len(hd_adj.intersection(body_adj)) / max_len)
    if(len(hd_adj)==0):
        return 0
    return(len(hd_adj.intersection(body_adj))/len(hd_adj))
