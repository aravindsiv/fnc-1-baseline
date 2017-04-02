





def common_noun_feature(hd_body_dict,max_len):
    hd_noun=set()
    body_noun=set()
    for i,k in enumerate(hd_body_dict["body_pos"]):
        if(k in ["PROPN","NOUN"]):
            if(hd_body_dict["body_tok"][i] in ["reports","report","reported","reportedly"] ):
                continue
            body_noun.add(hd_body_dict["body_tok"][i])
    for i,k in enumerate(hd_body_dict["hd_pos"]):
        if(k in ["PROPN","NOUN"]):
            if(hd_body_dict["hd_tok"][i] in ["reports","report","reported","reportedly"] ):
                continue
            hd_noun.add(hd_body_dict["hd_tok"][i])
    #return(len(hd_noun.intersection(body_noun))/len(hd_body_dict["hd_pos"]))
    return (len(hd_noun.intersection(body_noun)) / max_len)

def common_adjective_feature(hd_body_dict,max_len):
    hd_verb=set()
    body_verb=set()
    for i,k in enumerate(hd_body_dict["body_pos"]):
        if(k in ["ADV","VERB"]):
            if(hd_body_dict["body_tok"][i] in ["reports","report","reported","reportedly"] ):
                continue
            body_verb.add(hd_body_dict["body_tok"][i])

    for i,k in enumerate(hd_body_dict["hd_pos"]):
        if(k in ["ADV","VERB"]):
            if(hd_body_dict["hd_tok"][i] in ["reports","report","reported","reportedly"] ):
                continue
            hd_verb.add(hd_body_dict["hd_tok"][i])
    #return(len(hd_verb.intersection(body_verb))/len(hd_body_dict["hd_pos"]))
    return (len(hd_verb.intersection(body_verb)) / max_len)


def common_verb_feature(hd_body_dict,max_len):
    hd_adj=set()
    body_adj=set()
    for i,k in enumerate(hd_body_dict["body_pos"]):
        if(k in ["ADJ"]):
            body_adj.add(hd_body_dict["body_tok"][i])
    for i,k in enumerate(hd_body_dict["hd_pos"]):
        if(k in ["ADJ"]):
            hd_adj.add(hd_body_dict["hd_tok"][i])
    #return(len(hd_adj.intersection(body_adj))/len(hd_body_dict["hd_pos"]))
    return (len(hd_adj.intersection(body_adj)) / max_len)

