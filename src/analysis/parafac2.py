import os, sys
import numpy as np
import pickle
import argparse
from tensorly.decomposition import parafac2

sys.stdout.flush()
# print('stdout flushed')
            
def get_attributes():
    with open("./src/analysis/res/properties.lst", "r") as props:
        properties = props.readlines()
    return [prop.strip("\n") for prop in properties]

def get_lang_treebank_code():
    codes_dict = {} # {lang_code: treebank_name}
    with open("./src/analysis/res/languages_common_code.lst", "r") as f:
        line = f.readline()
        while line:
            treebank, code = line.strip().split()
            codes_dict[code] = treebank
            line = f.readline()
    return codes_dict

def load_and_process_data(langs_to_use, treebanks_root, filename, attribute, output_dir):
    lang_codes = get_lang_treebank_code()
    # These attributes, by default, have spaces but this just complicates a lot of file management.
    # So we replace attributes with spaces with others.
    if attribute not in get_attributes():
        raise ValueError("The provided attribute is not supported")
    attribute_old = attribute # retain this to be used later for filename
    if attribute == "POS":
        attribute = "Part of Speech"
    elif attribute == "ArgumentMark":
        attribute = "Argument Marking"
    elif attribute == "Gender":
        attribute = "Gender and Noun Class"
    elif attribute == "InfoStructure":
        attribute = "Information Structure"
    elif attribute == "Switch-Reference":
        attribute = "SwitchRef"
    
    for lang in langs_to_use:
        if lang not in lang_codes.keys():
            raise ValueError("The lang {} is not supported".format(lang))
        treebank = lang_codes[lang]
        data_path = os.path.join(treebanks_root, treebank, "{}-{}.pkl".format(treebank, filename))
        # print("Loading {} data".format(lang))
        data = pickle.load(open(data_path, 'rb'))
        # print("Data loaded")
        for l in range(0,13):
            data_filtered = []
            count = 0
            for d in data:
                if attribute in d['attributes'].keys():
                    data_filtered.append(d['layer_'+str(l)])
                    count += 1
            if len(data_filtered) != 0:
                # only proceed if the lang has this attributes
                data_filtered_np = np.vstack(data_filtered)
                if not os.path.exists(os.path.join(output_dir, treebank)):
                    os.makedirs(os.path.join(output_dir, treebank))
                np.save(os.path.join(output_dir, treebank, "{}-{}-LAYER-{}-ATTR-{}.npy".format(treebank, filename, l, attribute_old)), data_filtered_np)
            else:
                pass
            print("There are {} {} attributes in {} data".format(count, attribute, lang))


def PARAFAC2(langs_to_use, output_dir, filename_exp, filename_ctl, exp_name, layer, attribute, rank, verbose, n_iter_max):
    lang_codes = get_lang_treebank_code()
    # These attributes, by default, have spaces but this just complicates a lot of file management.
    # So we replace attributes with spaces with others.
    if attribute not in get_attributes() and attribute != "ZZZ":
        raise ValueError("The provided attribute is not supported")
    
    X = [] # stores all X_i in one list
    for lang in langs_to_use:
        if lang not in lang_codes.keys():
            raise ValueError("The lang {} is not supported".format(lang))
        treebank = lang_codes[lang]

        data_path_exp = os.path.join(output_dir, treebank, "{}-{}-LAYER-{}-ATTR-{}.npy".format(treebank, filename_exp, layer, attribute))
        if args.transliterate and lang in ["zh", "ja"]:
            # if we are using transliteration, we need to load the corresponding control model for zh and ja
            data_path_ctl = os.path.join(output_dir, treebank, "{}-roberta-base-TRANS-custom-mlm-pretrain-LAYER-{}-ATTR-{}.npy".format(treebank, layer, attribute))
        elif args.low_resource and lang == args.low_resource_lang and lang in ["en", "fr", "ko", "tr", "vi"]:
            # if we are using low-resource setting, we need to load the corresponding control model for en, fr, ko, tr, and vi
            if args.low_resource_10:
                data_path_ctl = os.path.join(output_dir, treebank, "{}-roberta-base-LOWRES10-custom-mlm-pretrain-LAYER-{}-ATTR-{}.npy".format(treebank, layer, attribute))
            else:
                data_path_ctl = os.path.join(output_dir, treebank, "{}-roberta-base-LOWRES-custom-mlm-pretrain-LAYER-{}-ATTR-{}.npy".format(treebank, layer, attribute))
        else:
            data_path_ctl = os.path.join(output_dir, treebank, "{}-{}-LAYER-{}-ATTR-{}.npy".format(treebank, filename_ctl, layer, attribute))
        
        try:
            acts_exp = np.load(data_path_exp) # data points x number of hidden dimension for exp model (e.g. exp model, multi-lingual model)
            acts_ctl = np.load(data_path_ctl) # data points x number of hidden dimension for ctl model (e.g. control model, mono-lingual model)
        except FileNotFoundError:
            print("The lang {} is does not has {} attribute".format(lang, attribute))
            continue

        # Mean subtract activations
        acts_exp_mean_subtracted = acts_exp - np.mean(acts_exp, axis=0, keepdims=True)
        acts_ctl_mean_subtracted = acts_ctl - np.mean(acts_ctl, axis=0, keepdims=True)

        X_i = np.matmul(acts_ctl_mean_subtracted.T, acts_exp_mean_subtracted) # ctl_model_hidden_dim x exp_model_hidden_dim

        # need to normalize it by dividing it by number of examples (on the cross-covariance matrix)
        assert acts_exp.shape[0] == acts_ctl.shape[0]
        X_i = X_i / acts_exp.shape[0] # divide by the number of data points 
        X.append(X_i)

    decompositions, errs = parafac2(X, rank=rank, init='svd', normalize_factors=False, return_errors=True, n_iter_max=n_iter_max, nn_modes=[0], verbose=verbose)
    (weights, factors, projections) = decompositions
    # weights 1D array (rank,)
    # factors: [ [len(X_i) x rank], [rank x rank], [exp_model_hidden_dim x rank] ] = [A, B, C]
    # projections: [ [ctl_model_hidden_dim x rank] x len(X_i) ] = [P_i]
    #              (there are len(X_i) such [ctl_model_hidden_dim x rank] matrices in projections)
    # B_i = P_i x B
    # X_i = B_i x diag(A[i]) x C.T
    save_dir = os.path.join(output_dir, 'decompositions', "{}-LAYER-{}-ATTR-{}.pickle".format(exp_name, layer, attribute))
    pickle.dump(decompositions, open(save_dir, 'wb'))
    # A, B, C = factors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument('--langs_to_use', nargs='+', type=str, default=["en", "ru", "zh", "ar", "hi"])
    parser.add_argument("--treebanks_root", type=str, default='./data/', help="Directory of treebanks")
    parser.add_argument("--output_dir", type=str, default='./data/', help="Directory of output for processed data")
    parser.add_argument("--filename", type=str, default='xlm-roberta-base-custom-mlm-pretrain', 
                        help="File name of the pkl file, only use this for preprocessing")
    parser.add_argument("--filename_exp", type=str, default='xlm-roberta-base-custom-mlm-pretrain', 
                        help="File name of the pkl file for the experimental model (e.g. a multilingual model), only use this for parafac2")
    parser.add_argument("--filename_ctl", type=str, default='roberta-base-custom-mlm-pretrain', 
                        help="File name of the pkl file for the control model (e.g. a monolingual model), only use this for parafac2")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--do_preprocess", action='store_true', 
                        help="Whether to do pre-process on data, i.e. get corresponding representations by attributes and layer")
    parser.add_argument("--do_parafac2", action='store_true', help="Whether to do PARAFAC2")
    parser.add_argument("--verbose", action='store_true', help="Whether to have verbose for PARAFAC2 computation")
    parser.add_argument("--layer", type=int, default=0, help="Layer of interest to do analysis, 0 is the embeding layer.")
    parser.add_argument("--rank", type=int, default=700, help="Rank used during PARAFAC2")
    parser.add_argument("--n_iter_max", type=int, default=100, help="Max iteration for PARAFAC2")
    parser.add_argument("--attribute", type=str, required=True, help="The attribute (aka. Unimorph dimension) \
                    to be probed (e.g., \"Number\", \"Gender\").", default=argparse.SUPPRESS)
    parser.add_argument("--transliterate", action="store_true", default=False, help="Whether to use transliterated data")
    parser.add_argument("--low_resource", action="store_true", default=False, help="Whether to use low resource data")
    parser.add_argument("--low_resource_10", action="store_true", default=False, help="Whether to use low resource data with 10 percent of the data")
    parser.add_argument("--low_resource_lang", type=str, default="en", help="The low resource language to use")
    args = parser.parse_args()
    
    if args.do_preprocess:
        load_and_process_data(args.langs_to_use, args.treebanks_root, args.filename, args.attribute, args.output_dir)
    if args.do_parafac2:
        PARAFAC2(args.langs_to_use, args.output_dir, args.filename_exp, args.filename_ctl, args.exp_name, args.layer,args.attribute, args.rank, args.verbose, args.n_iter_max)
    