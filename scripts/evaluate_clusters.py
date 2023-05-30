from sklearn import metrics
import numpy as np
import pickle as pkl

vis_pre_tsne, vis_fine_tsne, vis_pre_umap, vis_fine_umap = pkl.load(open('data/visualizations.pkl', 'rb'))
lab =  np.concatenate(tuple(pkl.load(open('data/wave2vec2_reco.pkl', 'rb')).values()), axis = 1)[0]
labs_pre, labs_pre_vis, labs_fine, labs_fine_vis = pkl.load(open("data/kmeans_umap_clusters","rb")) #kmeans_clusters for t-sne
cnn_pca, cnn_tsne, cnn_umap =  pkl.load(open('data/cnn_visualizations.pkl', 'rb'))
labs_cnn, labs_cnn_vis = pkl.load(open("data/cnn_part_kmeans_umap_clusters","rb"))

pre = pkl.load(open("data/wave2vec2_pretrained_embeddings.pkl", "rb"))
fine = pkl.load(open("data/wave2vec2_finetuned_embeddings.pkl", "rb"))
cnn = pkl.load(open("data/wave2vec2_finetuned_cnn_embeddings.pkl", "rb"))

pre_data = np.concatenate(tuple(pre.values()), axis=1)[0]
fine_data = np.concatenate(tuple(fine.values()), axis=1)[0]
cnn_data = np.concatenate(tuple(cnn.values()), axis=1)[0]


for f in [metrics.adjusted_mutual_info_score, metrics.mutual_info_score, metrics.homogeneity_score, metrics.completeness_score]:
    print(f)
    print("CNN features vs predictions")
    print(f(lab, labs_cnn))
    print("CNN features vs raw and fine clusters")
    print(f(labs_pre, labs_cnn))
    print(f(labs_fine, labs_cnn))
    
    print("Visualized and raw pre-trained feature clusterings:")
    print(f(lab, labs_pre_vis))
    print(f(lab, labs_pre))
    print("Comparison of raw and visualized:")
    print(f(labs_pre, labs_pre_vis))
    print("Visualized and raw fine-tuned feature clusterings:")
    print(f(lab, labs_fine_vis))
    print(f(lab, labs_fine))
    print("Comparison of raw and visualized:")
    print(f(labs_fine, labs_fine_vis))

    print("Comparison of pre and fine-tuned clusters:")
    print(f(labs_fine, labs_pre))
    print(f(labs_fine_vis, labs_pre_vis))

#print("Silhouette_scores (pre/fine):")
#print(metrics.silhouette_score(pre_data, labs_pre_vis, metric='cosine'))
#print(metrics.silhouette_score(pre_data, labs_pre, metric='cosine'))
#print(metrics.silhouette_score(fine_data, labs_fine_vis, metric='cosine'))
#print(metrics.silhouette_score(fine_data, labs_fine, metric='cosine'))
