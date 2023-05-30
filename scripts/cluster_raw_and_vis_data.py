#from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster import KMeans
from coclust.clustering.spherical_kmeans import SphericalKmeans

import pickle as pkl
import numpy as np
seed = 1024
import json
vis_pre_tsne, vis_fine_tsne, vis_pre_umap, vis_fine_umap = pkl.load(open('data/visualizations.pkl', 'rb'))
vis_pre_pca, vis_fine_pca = pkl.load(open('data/visualizations_pca.pkl', 'rb'))
pre = pkl.load(open("data/wave2vec2_pretrained_embeddings.pkl", "rb"))
fine = pkl.load(open("data/wave2vec2_finetuned_embeddings.pkl", "rb"))
pre_data = np.concatenate(tuple(pre.values()), axis=1)[0]
fine_data = np.concatenate(tuple(fine.values()), axis=1)[0]

#clustering_pre = OPTICS(metric = "cosine").fit(pre_data)
print("Clustering pretrained embeddings")
#clustering_pre = KMeans(n_clusters=36, random_state=0).fit(pre_data)
clustering_pre = SphericalKmeans(n_clusters=36, random_state=0)
clustering_pre.fit(pre_data)
#print(clustering_pre.cluster_hierarchy_)
labs_pre = clustering_pre.labels_


#clustering_pre_vis = KMeans(n_clusters=36, random_state=0).fit(vis_pre_pca)
clustering_pre_vis = SphericalKmeans(n_clusters=36, random_state=0)
clustering_pre_vis.fit(vis_pre_pca)
#print(clustering_pre.cluster_hierarchy_)
labs_pre_vis = clustering_pre_vis.labels_

vocab = json.load(open('finetuned_model/vocab.json'))
import matplotlib.pyplot as plt

from distinctipy import distinctipy

cmaps = distinctipy.get_colors(np.max(labs_pre)+1)
for i in range(np.max(labs_pre)+1):
    idx = np.where(labs_pre==i-1)
    plt.scatter(vis_pre_tsne[idx,0], vis_pre_tsne[idx,1], s=0.01, color =cmaps[i], label = "cluster_"+str(i-1))
    
    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol = 3, markerscale=16)
plt.savefig("pics/pre_pca_cluster_sph_kmeans.eps", bbox_inches = 'tight')
plt.close('all')
plt.figure().clear() 

#clustering_fine = OPTICS(metric = "cosine").fit(fine_data)

print("Clustering finetuned embeddings")
#clustering_fine = KMeans(n_clusters=36, random_state=0).fit(fine_data)
clustering_fine = SphericalKmeans(n_clusters=36, random_state=0)
clustering_fine.fit(fine_data)
labs_fine = clustering_fine.labels_

#clustering_fine_vis = KMeans(n_clusters=36, random_state=0).fit(vis_fine_pca)
clustering_fine_vis = SphericalKmeans(n_clusters=36, random_state=0)
clustering_fine_vis.fit(vis_fine_pca)
#print(clustering_pre.cluster_hierarchy_)
labs_fine_vis = clustering_fine_vis.labels_
#print(clustering_fine.cluster_hierarchy_)
pkl.dump([labs_pre, labs_pre_vis, labs_fine, labs_fine_vis], open("data/sph_kmeans_pca_clusters","wb"))
vocab = json.load(open('finetuned_model/vocab.json'))
import matplotlib.pyplot as plt

from distinctipy import distinctipy

cmaps = distinctipy.get_colors(np.max(labs_fine)+1)
for i in range(np.max(labs_fine)+1):
    idx = np.where(labs_fine==i-1)
    plt.scatter(vis_fine_tsne[idx,0], vis_fine_tsne[idx,1], s=0.01, color =cmaps[i], label = "cluster_"+str(i-1))
    
    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol = 3, markerscale=16)
plt.savefig("pics/fine_pca_cluster_sph_kmeans.eps", bbox_inches = 'tight')
plt.close('all')
plt.figure().clear() 
