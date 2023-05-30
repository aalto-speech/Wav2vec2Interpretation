import pickle as pkl
import numpy as np
import json, sys

vis_pre_tsne, vis_fine_tsne, vis_pre_umap, vis_fine_umap = pkl.load(open('data/visualizations.pkl', 'rb'))
lab_pred =  pkl.load(open('data/wave2vec2_reco.pkl', 'rb'))

def convert_to_id(text):
    t = text.split("/")
    id = t[-2]+t[-1].replace(".flac","")
    if id.startswith("ses"):
        id = t[-3]+id
    return id

def load_ref(file):
    with open(file,"rt",encoding="utf-8") as f:
        text=f.read().split('\n')
    ref = {}
    for t in text:
        ref[t.split(" ")[0]] = " ".join(t.split(" ")[1:])
    return ref

def load_metadata(file):
    with open(file,"rt",encoding="utf-8") as f:
        text=f.read().split('\n')
    meta = {}
    for t in text:
        meta[t.split(" ")[0]] = "none"
        if len(t.split(" "))>1:
            meta[t.split(" ")[0]] =t.split(" ")[1]
            if len(meta[t.split(" ")[0]])==0:
                meta[t.split(" ")[0]] = "none"
    return meta

#create legend ids and inv_vocab
meta_data = load_metadata(sys.argv[1])
meta_type = sys.argv[1].split("/")[-1]
inv_vocab = {}
vocab = {}
idx = 0
for val in set(meta_data.values()):
    inv_vocab[idx] = val
    vocab[val] = idx
    idx+=1
print(inv_vocab)
lab = []
for l in lab_pred:
    meta_lab = meta_data[convert_to_id(l)]
    lab+=[vocab[meta_lab]]*lab_pred[l].shape[1]
lab = np.array(lab)
import matplotlib.pyplot as plt


from distinctipy import distinctipy
legend = list(inv_vocab.values())
print(legend)
cmaps = distinctipy.get_colors(len(legend))
for i in range(len(legend)):
    idx = np.where(lab==i)
    print(i, np.sum(idx))
    plt.scatter(vis_pre_tsne[idx,0], vis_pre_tsne[idx,1], s=0.01, color =cmaps[i], label = inv_vocab[i])
    
    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol = 3, markerscale=16)
plt.savefig("pics/pre_tsne_"+meta_type +".eps", bbox_inches = 'tight')
plt.close('all')
plt.figure().clear() 


for i in range(len(legend)):
    idx = np.where(lab==i)
    plt.scatter(vis_fine_tsne[idx,0], vis_fine_tsne[idx,1], s=0.01, color =cmaps[i], label = inv_vocab[i])
    
    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol = 3, markerscale=16)
plt.savefig("pics/fine_tsne_"+meta_type +".eps", bbox_inches = 'tight')
plt.close('all')
plt.figure().clear() 

for i in range(len(legend)):
    idx = np.where(lab==i)
    plt.scatter(vis_pre_umap[idx,0], vis_pre_umap[idx,1], s=0.01, color =cmaps[i], label = inv_vocab[i])
    
    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol = 3, markerscale=16)
plt.savefig("pics/pre_umap_"+meta_type +".eps", bbox_inches = 'tight')
plt.close('all')
plt.figure().clear() 

for i in range(len(legend)):
    idx = np.where(lab==i)
    plt.scatter(vis_fine_umap[idx,0], vis_fine_umap[idx,1], s=0.01, color =cmaps[i], label = inv_vocab[i])
    
    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol = 3, markerscale=16)
plt.savefig("pics/fine_umap_"+meta_type +".eps", bbox_inches = 'tight')
plt.close('all')
plt.figure().clear() 


