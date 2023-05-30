from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA

import pickle as pkl
import numpy as np
import sys
import matplotlib.pyplot as plt

seed = 1024
pre = pkl.load(open("data/wave2vec2_pretrained_embeddings.pkl", "rb"))
fine = pkl.load(open("data/wave2vec2_finetuned_embeddings.pkl", "rb"))
cnn = pkl.load(open("data/wave2vec2_finetuned_cnn_embeddings.pkl", "rb"))

pre_data = np.concatenate(tuple(pre.values()), axis=1)[0]
fine_data = np.concatenate(tuple(fine.values()), axis=1)[0]
cnn_data = np.concatenate(tuple(cnn.values()), axis=1)[0]

pca = PCA(n_components=2)
cnn_pca = pca.fit_transform(cnn_data)

plt.scatter(cnn_pca[:,0], cnn_pca[:,1], s=0.01)
plt.savefig("pics/cnn_pca.svg")
plt.close('all')
plt.figure().clear() 


t_sne = TSNE(n_components=2,random_state=seed)
cnn_tsne = t_sne.fit_transform(cnn_data)
plt.scatter(cnn_tsne[:,0], cnn_tsne[:,1], s=0.01)
plt.savefig("pics/cnn_tsne.svg")
plt.close('all')
plt.figure().clear() 

umap_emb =umap.UMAP(n_components = 2, metric='cosine',random_state=seed)

cnn_umap = umap_emb.fit_transform(cnn_data)

pkl.dump([cnn_pca, cnn_tsne, cnn_umap], open('data/cnn_visualizations.pkl', 'wb'))

plt.scatter(cnn_umap[:,0], cnn_umap[:,1], s=0.01)
plt.savefig("pics/cnn_umap.svg")
plt.close('all')
plt.figure().clear() 


