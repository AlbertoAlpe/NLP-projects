import nltk 
from nltk.corpus import wordnet as wn 
import numpy as np 
import os 
from scipy.stats import spearmanr, pearsonr 
 
 
nltk.download('wordnet', quiet=True) 
nltk.download('omw-1.4', quiet=True) 
 
def wu_palmer_similarity(syn1, syn2): 
    lcs_candidates = syn1.lowest_common_hypernyms(syn2) 
    if not lcs_candidates: 
        return 0 
    lcs = lcs_candidates[0] 
    depth_lcs = lcs.max_depth() 
    depth_syn1 = syn1.max_depth() 
    depth_syn2 = syn2.max_depth() 
    return (2 * depth_lcs) / (depth_syn1 + depth_syn2) 
 
def shortest_path_similarity(syn1, syn2): 
    path_length = syn1.shortest_path_distance(syn2) 
    if path_length is None: 
        return 0 
    return 1 / (1 + path_length) 
 
def leacock_chodorow_similarity(syn1, syn2): 
    max_depth = wn.synset('entity.n.01').max_depth() 
    if max_depth == 0: 
        return 0 
    path_length = syn1.shortest_path_distance(syn2) 
    if path_length is None: 
        return 0 
    return -np.log((path_length + 1) / (2 * max_depth)) 
 
def max_similarity(term1, term2, similarity_function): 
    synsets1 = wn.synsets(term1) 
    synsets2 = wn.synsets(term2) 
     
    max_sim = 0 
     
    for syn1 in synsets1: 
        for syn2 in synsets2: 
            sim = similarity_function(syn1, syn2) 
            if sim > max_sim: 
                max_sim = sim 
    return max_sim 
 
def calculate_correlations(list, similarity_function): 
    similarities = [] 
    human_ratings = [float(row[2]) / 10 for row in combo_parole] 
     
    for row in list: 
        sim = max_similarity(row[0], row[1], similarity_function) 
        similarities.append(sim) 
         
    spearman_corr = spearmanr(human_ratings, similarities) 
    pearson_corr = pearsonr(human_ratings, similarities) 
    #print(human_ratings) 
    #print(similarities) 
    return spearman_corr, pearson_corr 
 
 
 
 
#current_dir = os.getcwd()
current_dir = os.path.dirname(__file__) 
file_name = 'WordSim353.csv'
 
file_path = os.path.join(current_dir, 'cc', file_name)

with open(file_name, 'r', encoding='utf-8') as train: 
   righe = train.readlines()[1:] 
 
combo_parole = [] 
 
for riga in righe: 
    riga = riga.strip().split(',') 
    combo_parole.append([riga[0], riga[1], riga[2]]) 
 
#print(combo_parole) 
     
 
spearman_wup, pearson_wup = calculate_correlations(combo_parole, wu_palmer_similarity) 
spearman_path, pearson_path = calculate_correlations(combo_parole, shortest_path_similarity) 
spearman_lch, pearson_lch = calculate_correlations(combo_parole, leacock_chodorow_similarity) 
 
print(f"Wu & Palmer - Spearman: {spearman_wup}, Pearson: {pearson_wup}") 
print(f"Shortest Path - Spearman: {spearman_path}, Pearson: {pearson_path}") 
print(f"Leacock & Chodorow - Spearman: {spearman_lch}, Pearson: {pearson_lch}")