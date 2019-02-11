import numpy as np
import sys
import argparse
import os
import string
import re
import json

FPP_1003824025 = open("/u/cs401/Wordlists/First-person", "r").read().split()
FPP_1003824025 = '( |^)' + '/|( |^)'.join(FPP_1003824025)
SPP_1003824025 = open("/u/cs401/Wordlists/Second-person", "r").read().split()
SPP_1003824025 = '( |^)' + '/|( |^)'.join(SPP_1003824025)
TPP_1003824025 = open("/u/cs401/Wordlists/Third-person", "r").read().split()
TPP_1003824025 = '( |^)' + '/|( |^)'.join(TPP_1003824025)
CON_1003824025 = open("/u/cs401/Wordlists/Conjunct", "r").read().split()
CON_1003824025 = '( |^)' + '/|( |^)'.join(CON_1003824025)
SLANG_1003824025 = open("/u/cs401/Wordlists/Slang", "r").read().split()
SLANG_1003824025 = '( |^)' + '/|( |^)'.join(SLANG_1003824025)

BN_GL_1003824025 = open("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv",
                        "r").read().split()
BN_GL_1003824025 = [row.split(",") for row in BN_GL_1003824025]
BN_GL_1003824025 = BN_GL_1003824025[1: len(BN_GL_1003824025) -1]
RW_1003824025 = open("/u/cs401/Wordlists/Ratings_Warriner_et_al.csv",
	                     "r").readlines()[1:]
RW_1003824025 = [row.split(",") for row in RW_1003824025]
RW_WORD_TO_ROW_1003824025 = {}
BN_WORD_TO_ROW_1003824025 = {}
for row in BN_GL_1003824025:
    BN_WORD_TO_ROW_1003824025[row[1]] = row
for row in RW_1003824025:
    RW_WORD_TO_ROW_1003824025[row[1]] = row

ALT_FEAT_1003824025 = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy', "r")
ALT_ID_1003824025 = open("/u/cs401/A1/feats/Alt_IDs.txt", "r").read().split()
LEFT_FEAT_1003824025 = np.load('/u/cs401/A1/feats/Left_feats.dat.npy', "r")
LEFT_ID_1003824025 = open("/u/cs401/A1/feats/Left_IDs.txt", "r").read().split()
RIGHT_FEAT_1003824025 = np.load('/u/cs401/A1/feats/Right_feats.dat.npy', "r")
RIGHT_ID_1003824025 = open("/u/cs401/A1/feats/Right_IDs.txt", "r").read().split()
CENTER_FEAT_1003824025 = np.load('/u/cs401/A1/feats/Center_feats.dat.npy', "r")
CENTER_ID_1003824025 = open("/u/cs401/A1/feats/Center_IDs.txt", "r").read().split()
FEAT_DATA_1003824025 = {"Alt": (ALT_FEAT_1003824025, ALT_ID_1003824025),
                        "Left": (LEFT_FEAT_1003824025, LEFT_ID_1003824025),
                        "Right": (RIGHT_FEAT_1003824025, RIGHT_ID_1003824025),
                        "Center": (CENTER_FEAT_1003824025, CENTER_ID_1003824025)}
CAT_NUM_1003824025 = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}


def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    upper = len(re.compile(r"[A-Z][A-Z][A-Z]+/").findall(comment))
    comment = comment.lower()
    fpp = len(re.compile(r"" + FPP_1003824025, re.IGNORECASE).findall(comment))
    spp = len(re.compile(r"" + SPP_1003824025, re.IGNORECASE).findall(comment))
    tpp = len(re.compile(r"" + TPP_1003824025, re.IGNORECASE).findall(comment))
    con = len(re.compile(r"" + CON_1003824025, re.IGNORECASE).findall(comment))
    vbd = len(re.compile(r"/vbd").findall(comment))
    ft_regex = "'ll|will|gonna"
    ftv = len(re.compile(r"" + ft_regex).findall(comment)) + len(
        re.compile(r"go/vbg to/in [a-z]+/vb").findall(comment))
    comma = len(re.compile(r"" + ",/,").findall(comment))
    punc = len(re.compile(r"[?!.]+/[.]+ [?!.]+/[.]+").findall(
        comment))

    cnouns = len(re.compile(r"(\/nns |\/nn )").findall(comment))
    pnouns = len(re.compile(r"(\/nnp |\/nnps )").findall(comment))
    adverbs = len(re.compile(r"(\/rb |\/rbr |\/rbs )").findall(comment))
    wh = len(re.compile(r"(\/wdt |\/wp |\/wp\$ |\/wrb )").findall(comment))
    slang = len(re.compile(r"" + SLANG_1003824025, re.IGNORECASE).findall(comment))
    tokens = []
    aoa_scores = []
    img_scores = []
    fam_scores = []
    vms_scores = []
    ams_scores = []
    dms_scores = []
    for tg in comment.split(" "):
        tg = tg.split("/")
        if len(tg) > 1:
            token = tg[0]
            if token != '' and token not in string.punctuation:
                tokens.append(token)
            if token != '' and token in BN_WORD_TO_ROW_1003824025:
                aoa_scores.append(float(BN_WORD_TO_ROW_1003824025[token][3]))
                img_scores.append(float(BN_WORD_TO_ROW_1003824025[token][4]))
                fam_scores.append(float(BN_WORD_TO_ROW_1003824025[token][5]))
            if token != '' and token in RW_WORD_TO_ROW_1003824025:
                vms_scores.append(float(RW_WORD_TO_ROW_1003824025[token][2]))
                ams_scores.append(float(RW_WORD_TO_ROW_1003824025[token][5]))
                dms_scores.append(float(RW_WORD_TO_ROW_1003824025[token][8]))

    avg_word_len = sum(len(word) for word in tokens) / len(tokens) if len(
        tokens) > 0 else 0
    num_sent = len(comment.split("\n"))
    avg_sent_leng = sum(len(word.split(" ")) for word in comment.split(
        "\n"))/num_sent
    avg_aoa = sum(aoa_scores)/len(aoa_scores) if len(aoa_scores) > 0 else 0
    avg_img = sum(img_scores)/len(img_scores) if len(img_scores) > 0 else 0
    avg_fam = sum(fam_scores) / len(fam_scores) if len(fam_scores) > 0 else 0
    std_aoa = np.std(aoa_scores) if len(aoa_scores) > 0 else 1
    std_img = np.std(img_scores) if len(img_scores) > 0 else 1
    std_fam = np.std(fam_scores) if len(fam_scores) > 0 else 1
    avg_vms = sum(vms_scores)/len(vms_scores) if len(vms_scores) > 0 else 0
    avg_ams = sum(ams_scores)/len(ams_scores) if len(ams_scores) > 0 else 0
    avg_dms = sum(dms_scores)/len(dms_scores) if len(dms_scores) > 0 else 0
    std_vms = np.std(vms_scores) if len(vms_scores) > 0 else 1
    std_ams = np.std(ams_scores) if len(ams_scores) > 0 else 1
    std_dms = np.std(dms_scores) if len(dms_scores) > 0 else 1

    feats = np.zeros(173)
    feats[0] = fpp
    feats[1] = spp
    feats[2] = tpp
    feats[3] = con
    feats[4] = vbd
    feats[5] = ftv
    feats[6] = comma
    feats[7] = punc
    feats[8] = cnouns
    feats[9] = pnouns
    feats[10] = adverbs
    feats[11] = wh
    feats[12] = slang
    feats[13] = upper
    feats[14] = avg_sent_leng
    feats[15] = avg_word_len
    feats[16] = num_sent
    feats[17] = avg_aoa
    feats[18] = avg_img
    feats[19] = avg_fam
    feats[20] = std_aoa
    feats[21] = std_img
    feats[22] = std_fam
    feats[23] = avg_vms
    feats[24] = avg_ams
    feats[25] = avg_dms
    feats[26] = std_vms
    feats[27] = std_ams
    feats[28] = std_dms
    return feats




def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    index = 0
    for d in data:
        cat = d["cat"]
        id_ = d["id"]
        extract_feats = extract1(d["body"])
        liwc_index = FEAT_DATA_1003824025[cat][1].index(id_)
        liwc = FEAT_DATA_1003824025[cat][0][liwc_index]
        cat_np = np.array([CAT_NUM_1003824025[cat]])
        for i in range(29, 173):
            extract_feats[i] = liwc[i-29]
        result = np.concatenate([extract_feats, cat_np])
        feats[index] = result
        index += 1

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
    main(args)
