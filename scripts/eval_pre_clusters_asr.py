import numpy as np
import pickle as pkl
import json
import librosa
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
)

loc = "./finetuned_model/" #'facebook/wav2vec2-large-100k-voxpopuli'
processor =  Wav2Vec2Processor.from_pretrained(loc, cache_dir = "tmp")
vocab = json.load(open('finetuned_model/vocab.json'))

inv_vocab = {v: k for k, v in vocab.items()}

np.set_printoptions(threshold=50)
vis_pre_tsne, vis_fine_tsne, vis_pre_umap, vis_fine_umap = pkl.load(open('data/visualizations.pkl', 'rb'))
lab =  np.concatenate(tuple(pkl.load(open('data/wave2vec2_reco.pkl', 'rb')).values()), axis = 1)[0]
reco_lab = pkl.load(open('data/wave2vec2_reco.pkl', 'rb'))
labs_pre, labs_pre_vis, labs_fine, labs_fine_vis = pkl.load(open("data/kmeans_clusters","rb"))
labs_cnn, labs_cnn_vis = pkl.load(open("data/cnn_part_kmeans_umap_clusters","rb"))

pre = pkl.load(open("data/wave2vec2_pretrained_embeddings.pkl", "rb"))
fine = pkl.load(open("data/wave2vec2_finetuned_embeddings.pkl", "rb"))
pre_data = np.concatenate(tuple(pre.values()), axis=1)[0]
fine_data = np.concatenate(tuple(fine.values()), axis=1)[0]

discovered_map = {}
for idx in inv_vocab:
    data = labs_pre[lab==idx]
    distr,_ = np.histogram(data, list(inv_vocab.keys()))
    max_category = np.argmax(distr)
    if max_category in discovered_map:
        discovered_map[max_category].append(inv_vocab[idx])
    else:
        discovered_map[max_category] = [inv_vocab[idx]]
    percent = distr[max_category]/np.sum(distr)
    print("cluster for "+inv_vocab[idx]+": "+str(max_category)+ ", " + str(percent) +"%" )
    print(distr)
print(discovered_map)

mapping = {}
for idx in range(np.max(labs_pre)+1):
    data = lab[labs_pre==idx]
    distr,_ = np.histogram(data, list(np.arange(np.max(labs_pre))))
    print(distr)
    max_category = np.argmax(distr[1:])
    mapping[idx] = max_category
print(mapping)

idx = 0
transcr = {}
for utt in reco_lab:
    end = idx+reco_lab[utt].shape[-1]
    corrected = [mapping[clus] for clus in labs_pre[idx:end]]
    print(utt)
    text = "".join(processor.batch_decode(corrected)).replace("<unk>","")
    transcr[utt] = ""
    for c in text:
        if len(transcr[utt])<1 or transcr[utt][-1] != c:
            transcr[utt] +=c
    print(transcr[utt])
    idx = end
    
pkl.dump(transcr,open("data/wav2vec2_clustered_pre_transcr.pkl","wb"))

