import pympi
import pandas as pd
from argparse import ArgumentParser
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
import numpy as np
import string
import os

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-e", "--eaf", type=str, nargs="+",
                        help="Target eaf file.")
    parser.add_argument("-o", "--output", type=str,
                        default="kc_killkan-ud-test.conllu",
                        help="Output conllu file.")
    args = parser.parse_args()
    if args.eaf == ["all"]:
        args.eaf = [f"Chapter{i}/Chapter{i}.eaf" for i in range(1, 21)]
    return args

def generate_list_per_tier(eaf: pympi.Elan.Eaf,
                           tier: str) -> List[list]:
    """Generate a list of elements per tier.
    args:
    - eaf (pympi.Elan.Eaf): an eaf file (read by pympi)
    - tier (str): tier name
    return:
    - a list of lists. Each list corresponds to a sentence.
    """
    content = eaf.get_annotation_data_for_tier(tier)
    elem_list = []
    text = ""
    time = 0
    for c in content:
        elem = c[2]
        text_ref = c[3]
        time_ref = c[0]
        # if text != text_ref: # new sentence
        if time != time_ref: # new sentence
            elem_list.append([])
        elem_list[-1].append(elem)
        # text = text_ref
        time = time_ref
    return elem_list

def misc_to_dict(misc: str) -> dict:
    """Convert a MISC annotation (str) into a dict.
    Also applicable to FEATS."""
    misc_items = misc.split("|")
    misc_dict = dict()
    for m in misc_items:
        if m == "":
            continue
        key, value = m.split("=")
        misc_dict[key] = value
    return misc_dict

def dict_to_misc(misc_dict: dict) -> str:
    """Convert a MISC dict back to a str.
    Also applicable to FEATS."""
    misc_list = []
    for k, v in misc_dict.items():
        misc_list.append(k + "=" + v)
    misc = "|".join(misc_list)
    return misc

def generate_conllu_list(eaf: pympi.Elan.Eaf,
                         chapter_id: str) -> List[List[dict]]:
    """Generate a UD-convertable list from eaf.
    args:
    - eaf (pympi.Elan.Eaf): an eaf file (read by pympi)
    - chapter_id (str): chapter ID
    return:
    - List[List[dict]]: The first layer of list corresponds to each sentence.
    Each element of the list is another layer of list containing dicts.
    Each dict corresponds to a row in the UD format (i.e., word).
    """
    sentences = eaf.get_annotation_data_for_tier("default")
    translations = eaf.get_annotation_data_for_tier("Spanish")
    id_list = generate_list_per_tier(eaf, "ID")
    form_list = generate_list_per_tier(eaf, "FORM")
    lemma_list = generate_list_per_tier(eaf, "LEMMA")
    upos_list = generate_list_per_tier(eaf, "UPOS")
    feats_list = generate_list_per_tier(eaf, "FEATS")
    head_list = generate_list_per_tier(eaf, "HEAD")
    deprel_list = generate_list_per_tier(eaf, "DEPREL")
    misc_list = generate_list_per_tier(eaf, "MISC")
    conllu = []
    assert len(sentences) == len(translations), print(len(sentences), len(translations), "Chapter", chapter_id)
    assert len(sentences) == len(id_list), print(len(sentences), len(id_list), id_list, "Chapter", chapter_id)
    for i, sent in enumerate(sentences):
        entry = dict()
        sent_id = str(chapter_id) + "_" + str(i)
        entry["sent_id"] = sent_id
        entry["sent"] = sent[2]
        try:
            entry["trans"] = translations[i][2]
        except ValueError as e:
            print(e)
            print(translations[i])
            print(f"Error found in Chapter {chapter_id} Sent {sent_id} Sent {sent[2]}")
        except IndexError as ie:
            print(ie)
            print(f"Error found in Chapter {chapter_id} Sent {sent_id} Sent: {sent[2]}")
            print("Index:", i)
            print("Number of translations:", len(translations))
        entry["start"] = sent[0]
        entry["end"] = sent[1]
        entry["annotation"] = []
        try:
            ids = id_list[i]
            ids = [int(idx) if "-" not in idx else sum(map(int, idx.split("-")))/2-1 for idx in ids]
        except ValueError as ve:
            print(ve)
            print(f"Error found in Chapter {chapter_id} Sent {sent_id} Sent {sent[2]}")
        except IndexError as ie:
            print(ie)
            print(f"Error found in Chapter {chapter_id} Sent {sent_id} Sent {sent[2]}")
        try:
            forms = form_list[i]
        except IndexError as ie:
            print(ie)
            print(f"Error found in Chapter {chapter_id} Sent {sent_id} Sent {sent[2]}")
        lemmas = lemma_list[i]
        uposs = upos_list[i]
        xposs = ["_" for _ in range(len(id_list[i]))]
        featss = feats_list[i]
        heads = head_list[i]
        deprels = deprel_list[i]
        relss = ["_" for _ in range(len(id_list[i]))]
        miscs = misc_list[i]

        elems = zip(ids, forms, lemmas, uposs, xposs,
                    featss, heads, deprels, relss, miscs)
        elems = sorted(elems)

        for j, elem in enumerate(elems):
            row = dict()
            idx = elem[0]
            if type(idx) == float:
                idx = str(int(idx) + 1) + "-" + str(int(idx) + 2)
            else:
                idx = str(idx)
            row["ID"] = idx

            form = forms[j]
            row["FORM"] = form

            lemma = elem[2]
            if not lemma:
                lemma = "_"
            row["LEMMA"] = lemma

            upos = elem[3]
            if not upos:
                upos = "_"
            row["UPOS"] = upos

            xpos = elem[4]
            if not xpos:
                xpos = "_"
            row["XPOS"] = xpos

            feats = elem[5]
            if not feats:
                feats = "_"
            row["FEATS"] = feats

            head = elem[6]
            if not head:
                head = "_"
            row["HEAD"] = head

            deprel = elem[7]
            if not deprel:
                deprel = "_"
            row["DEPREL"] = deprel

            rels = elem[8]
            if not rels:
                rels = "_"
            row["RELS"] = rels

            misc = elem[9]
            try:
                misc_dict = misc_to_dict(misc)
            except ValueError as e:
                print(e)
                print(f"Error in: Chapter {chapter_id} Sent {sent_id}")
                print(misc)
            if j < len(elems) - 1:
                if elems[j+1][1] in string.punctuation:
                    misc_dict["SpaceAfter"] = "No"
                    misc_dict = dict(sorted(misc_dict.items())) # sort by alphabetical order
                    misc = dict_to_misc(misc_dict)
            if "CSID" not in misc_dict.keys():
                # Kichwa
                misc_dict["CSID"] = "KC"
                misc_dict["Lang"] = "qu"
                misc_dict = dict(sorted(misc_dict.items())) # sort by alphabetical order
                misc = dict_to_misc(misc_dict)
            row["MISC"] = misc
            entry["annotation"].append(row)
        conllu.append(entry)
    return conllu

def generate_conllu_str(entry: dict) -> str:
    """Generate a CoNLLU-style text per sentence.
    """
    content = f"# sent_id = {entry['sent_id']}\n"
    content += f"# sent = {entry['sent']}\n"
    content += f"# trans = {entry['trans']}\n"
    content += f"# start = {entry['start']}\n"
    content += f"# end = {entry['end']}\n"
    
    for row in entry["annotation"]:
        line = row["ID"] + "\t" + row["FORM"] + "\t" + \
            row["LEMMA"] + "\t" + row["UPOS"] + "\t" + \
            row["XPOS"] + "\t" + row["FEATS"] + "\t" + \
            row["HEAD"] + "\t" + row["DEPREL"] + "\t" + \
            row["RELS"] + "\t" + row["MISC"] + "\n"
        content += line
    return content

def get_tiers(eaf: pympi.Elan.Eaf):
    tier_dict = dict()
    default_tier = eaf.tiers["default"][0]
    tier_dict["default"] = default_tier

    child_tiers = ["ID", "FORM", "LEMMA", "UPOS", "FEATS",
                   "HEAD", "DEPREL", "MISC"]
    for t in child_tiers:
        tier_dict[t] = eaf.tiers[t][1]
    return tier_dict

def get_values(tier_content: dict) -> list:
    """Get values included in the tier."""
    values = [e[1] for e in tier_content.values()]
    return values

class ELAN2UD:
    def __init__(self, eaf: pympi.Elan.Eaf):
        self.eaf = eaf
        self.sent_tier = eaf.tiers["default"][0]
        self.trans_tier = eaf.tiers["Spanish"][1]
        self.form_tier = eaf.tiers["FORM"][1]
        self.id_tier = eaf.tiers["ID"][1]
        self.lemma_tier = eaf.tiers["LEMMA"][1]
        self.upos_tier = eaf.tiers["UPOS"][1]
        self.feats_tier = eaf.tiers["FEATS"][1]
        self.head_tier = eaf.tiers["HEAD"][1]
        self.deprel_tier = eaf.tiers["DEPREL"][1]
        self.misc_tier = eaf.tiers["MISC"][1]

    def get_text_translation(self):
        """Get text and translation."""
        translations = [a[1] for a in self.trans_tier.values()]
        text = [a[2] for a in self.sent_tier.values()]
        assert len(translations) == len(text)
        kichwa_spanish = list(zip(text, translations))
        return kichwa_spanish

    def get_kichwa_spanish_csv(self, kichwa_spanish: List[Tuple],
                               dst="kichwa_spanish.csv") -> None:
        """Create a csv file containing the aligned Kichwa sentences
        and their Spanish translations.                                                                                                                                                                                                            
        Arguments:
        - `kichwa_spanish`: a list of tuples containing a Kichwa
        sentence and its Spanish translation.
        - `dst` (str): the path to the ouptut csv file for the
        dataframe to be saved.
        """
        df = pd.DataFrame(kichwa_spanish, columns=["Kichwa", "Spanish"])
        
        # Copy the Spanish column to keep track of the change.
        df["Original"] = df["Spanish"]
        df.to_csv(dst)

class MorphComplex:
    def __init__(self, eaf: pympi.Elan.Eaf):
        self.eaf = eaf
        self.sent_tier = eaf.tiers["default"][0]
        self.form_tier = eaf.tiers["FORM"][1]
        self.id_tier = eaf.tiers["ID"][1]
        self.lemma_tier = eaf.tiers["LEMMA"][1]
        self.upos_tier = eaf.tiers["UPOS"][1]
        self.feats_tier = eaf.tiers["FEATS"][1]
        self.head_tier = eaf.tiers["HEAD"][1]
        self.deprel_tier = eaf.tiers["DEPREL"][1]
        self.misc_tier = eaf.tiers["MISC"][1]

        # Store the entire corpus information in a dict of DataFrames.
        self.df_dict = self.eaf2df_all()

    def get_sent_sentid(self) -> List[Tuple[str, str]]:
        """Get a list of tuples of sentence-id pair."""
        id_sent_pair = [(k, v[2]) for k, v in self.sent_tier.items()]
        return id_sent_pair

    def get_sentid_forms(self) -> defaultdict:
        """Get a list of FORMs (tokens) in the text with `sentid`.
        For example, `sentid_form_dict["a1"] would look like ...
        [('a405', 'Ari'),
        ('a406', ','),
        ('a407', 'ari'), ...]
        """
        sentid_form_list = [(k, v[0], v[1]) for k, v in self.form_tier.items()]
        sentid_form_dict = defaultdict(lambda: list())
        for form_id, a_id, form in sentid_form_list:
            sentid_form_dict[a_id].append((form_id, form))
        return sentid_form_dict

    def get_childtiers_dict(self):
        """Convert the child tiers (ID, LEMMA, UPOS, FEATS, DEPREL,
        MISC) into dictionaries, whose keys are FORM annotation id.
        For example, `childtiers["LEMMA"] would look like...
        {'a405': 'ari',
        'a406': ',',
        'a407': 'ari', ... }
        """
        formid_lemma_dict = {v[0]: v[1] for v in self.lemma_tier.values()}
        formid_id_dict = {v[0]: v[1] for v in self.id_tier.values()}
        formid_upos_dict = {v[0]: v[1] for v in self.upos_tier.values()}
        formid_feats_dict = {v[0]: v[1] for v in self.feats_tier.values()}
        formid_head_dict = {v[0]: v[1] for v in self.head_tier.values()}
        formid_deprel_dict = {v[0]: v[1] for v in self.deprel_tier.values()}
        formid_misc_dict = {v[0]: v[1] for v in self.misc_tier.values()}
        childtiers_dict = {"LEMMA": formid_lemma_dict,
                           "ID": formid_id_dict,
                           "UPOS": formid_upos_dict,
                           "FEATS": formid_feats_dict,
                           "HEAD": formid_head_dict,
                           "DEPREL": formid_deprel_dict,
                           "MISC": formid_misc_dict}
        return childtiers_dict
    
    def eaf2df_sent(self, sentid: str):
        """Convert eaf (per sentence) into UD-style DataFrame."""
        sentid_form_dict = self.get_sentid_forms()
        childtiers_dict = self.get_childtiers_dict()
        ud_table = []
        for formid, form in sentid_form_dict[sentid]:
            try:
                lemma = childtiers_dict["LEMMA"][formid]
                id_ = childtiers_dict["ID"][formid]
                upos = childtiers_dict["UPOS"][formid]
                feats = childtiers_dict["FEATS"][formid]
                head = childtiers_dict["HEAD"][formid]
                deprel = childtiers_dict["DEPREL"][formid]
                misc = childtiers_dict["MISC"][formid]
                column = [id_, form, lemma, upos, feats, head, deprel, misc]
                ud_table.append(column)
            except KeyError as e:
                print("KeyError:", e)
        df = pd.DataFrame(ud_table,
                          columns=["ID", "FORM", "LEMMA", "UPOS",
                                   "FEATS", "HEAD", "DEPREL", "MISC"])
        return df

    def eaf2df_all(self) -> Dict[str, pd.DataFrame]:
        """Convert the entire document to a dictionary of DataFrames
        per sentence, whose keys are sent_ids."""
        sentid_sent_pairs = self.get_sent_sentid()
        print(sentid_sent_pairs)
        """ ->
        [('a1', 'Ari, ari, kikinkuna, wawkikuna panikuna.'),
        ('a171', 'Kayman, kayman shamuychik.'),
        ... ]"""
        sentid_sent_dict = {s[0]: s[1] for s in sentid_sent_pairs}
        sentid_form_dict = self.get_sentid_forms()
        print(sentid_form_dict)
        df_list = [] # List of sentence-level dataframes
        df_dict = dict()
        for sentid, form in sentid_form_dict.items():
            df = self.eaf2df_sent(sentid)
            
            sent_table_dict = dict()
            sent_table_dict["sent"] = sentid_sent_dict[sentid]
            sent_table_dict["df"] = df
            # TODO: add the Spanish tier
            df_dict[sentid] = sent_table_dict
        return df_dict

    def apply_lower(self, df_dict: dict, tier_name: str) -> dict:
        """Apply .lower() to all the elements in the tier.
        """
        for sent_id, sent_table_dict in df_dict.items():
            df = sent_table_dict["df"]
            df[tier_name].map(lambda elem: elem.lower())
            df_dict[sent_id]["df"] = df
        return df_dict

    def filter_num(self, df_dict: dict):
        for sent_id, sent_table_dict in df_dict.items():
            df = sent_table_dict["df"]
            df = df[df["UPOS"] != "NUM"]
            df_dict[sent_id]["df"] = df
        return df_dict

    def filter_pos(self, df_dict: dict, filter_pos: set):
        for sent_id, sent_table_dict in df_dict.items():
            df = sent_table_dict["df"]
            df = df[~df["UPOS"].isin(filter_pos)]
            df_dict[sent_id]["df"] = df
        return df_dict
    
    def get_ttr(self,
                lowercase=True,
                filter_num=True,
                filter_pos={"X", "PUNCT"}):
        """Compute the Type-Token Ratio from EAF.
        """
        df_dict = self.df_dict
        if lowercase:
            df_dict = self.apply_lower(df_dict, "FORM")
        if filter_num:
            df_dict = self.filter_num(df_dict)
        if filter_pos:
            df_dict = self.filter_pos(df_dict, filter_pos)

        # List up all the FORMs
        forms = []
        for sent_table_dict in df_dict.values():
            forms += list(sent_table_dict["df"]["FORM"])
                
        return len(set(forms)) / len(forms)

    def get_msp(self,
                lowercase=True,
                filter_num=True,
                filter_pos={"X", "PUNCT"}):
        """Compute the Mean Size of Paradigms.
        MSP = len(set(T)) / len(set(L))
        """
        df_dict = self.df_dict
        if lowercase:
            df_dict = self.apply_lower(df_dict, "LEMMA")
            df_dict = self.apply_lower(df_dict, "FORM")
        if filter_num:
            df_dict = self.filter_num(df_dict)
        if filter_pos:
            df_dict = self.filter_pos(df_dict, filter_pos)

        # List up all the FORMs and LEMMAs
        forms = []
        lemmas = []
        for sent_table_dict in df_dict.values():
            forms += list(sent_table_dict["df"]["FORM"])
            lemmas += list(sent_table_dict["df"]["LEMMA"])

        return len(set(forms)) / len(set(lemmas))

    def get_wh_lh(self,
                  lowercase=True,
                  filter_num=True,
                  filter_pos={"X", "PUNCT"}):
        """Compute the Word Entropy and Lemma Entropy.
        WH = - sum(p(t_i) * log(p(t_i)))
        LH = - sum(p(l_i) * log(p(l_i)))
        """
        df_dict = self.df_dict
        if lowercase:
            df_dict = self.apply_lower(df_dict, "FORM")
            df_dict = self.apply_lower(df_dict, "LEMMA")
        if filter_num:
            df_dict = self.filter_num(df_dict)
        if filter_pos:
            df_dict = self.filter_pos(df_dict, filter_pos)

        # List up all the FORMs/LEMMAs and count them
        forms = []
        lemmas = []
        for sent_table_dict in df_dict.values():
            forms += list(sent_table_dict["df"]["FORM"])
            lemmas = list(sent_table_dict["df"]["LEMMA"])

        # total
        num_forms = len(forms)
        num_lemmas = len(lemmas)

        # frequency
        forms_freq = Counter(forms)
        lemmas_freq = Counter(lemmas)

        wh, lh = 0, 0
        for w in forms_freq:
            p = forms_freq[w] / num_forms
            wh -= p * np.log2(p)
        for l in lemmas_freq:
            p = lemmas_freq[l] / num_lemmas
            lh -= p * np.log2(p)
        return wh, lh

    def get_mfh(self,
                filter_num=True,
                filter_pos={"X", "PUNCT"}):
        """Compute the Morphological Features Entropy.
        MFH = - sum(p(phi) * log(p(phi)))
        """
        if filter_num:
            df_dict = self.filter_num(df_dict)
        if filter_pos:
            df_dict = self.filter_pos(df_dict, filter_pos)

        feats_num = 0
        feats = []
        for sent_table_dict in df_dict.values():
            feat = list(sent_table_dict["df"]["FEATS"]) # ["Polarity=Neg", "Person=3|Tense=Sing|VerbForm=Fin", ""]
            feat = [f for f in feat if f != ""]
            for f in feat:
                feats += f.split("|")
        feats_freq = Counter(feats) # Frequency of features
        feats_num = sum(feats_freq.values()) # Number of features
        for f in feats_freq:
            p = feats_freq[f] / feats_num
            mfh -= p * np.log2(p)
        return mfh

    def get_is(self):
        """Compute the Inflectional Synthesis.
        IS = len(set(Phi))
        """
        raise NotImplementedError
            
if __name__ == "__main__":
    args = get_args()

    ud_str = ""
    for eaf_file in args.eaf:
        chapter_id = os.path.splitext(os.path.split(eaf_file)[1])[0][7:]
        eaf = pympi.Eaf(eaf_file)
        ud_list = generate_conllu_list(eaf, chapter_id)
        for sent in ud_list:
            ud_str += generate_conllu_str(sent) + "\n"

    with open(args.output, "w") as f:
        f.write(ud_str)
