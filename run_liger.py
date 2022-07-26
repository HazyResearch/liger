import sys
sys.path.append('/hdd2/dyah/liger/liger')

import os
import torch
import argparse

import numpy as np
from sklearn.cluster import KMeans

from liger import Liger
from flyingsquid_cluster import Flyingsquid_Cluster

from core import load_config
from utils import evaluate_models, test_model, evaluate_thresholds, cluster_embeddings

engine = "ada"
dataset = "spam"
embedding_path = f"/hdd2/dyah/epoxy-clip/artifacts/OpenAI/{engine}/{dataset}_records"

# Load L and Y matrices
L_dev_raw_orig = torch.load(os.path.join(embedding_path, 'val_L.pt')).detach().cpu().numpy()
Y_dev_raw = torch.load(os.path.join(embedding_path, 'val_Y.pt')).detach().cpu().numpy()
Y_test_raw = torch.load(os.path.join(embedding_path, 'test_Y.pt')).detach().cpu().numpy()
L_train_raw_orig = torch.load(os.path.join(embedding_path, 'train_L.pt')).detach().cpu().numpy()
L_test_raw_orig = torch.load(os.path.join(embedding_path, 'test_L.pt')).detach().cpu().numpy()
Y_train_raw = torch.load(os.path.join(embedding_path, 'train_Y.pt')).detach().cpu().numpy()

avg_embeddings_train = torch.load(os.path.join(embedding_path, 'train_feature.pt')).detach().cpu().numpy()
avg_embeddings_dev = torch.load(os.path.join(embedding_path,'val_feature.pt')).detach().cpu().numpy()
avg_embeddings_test = torch.load(os.path.join(embedding_path,'test_feature.pt')).detach().cpu().numpy()


def main(args):
    cfg = load_config(args.config)
    thresholds = cfg['thresholds']
    n_clusters = cfg['n_clusters']
    
    liger = Liger()
    L_train_expanded = liger.expand_lfs(
        L_train_raw_orig, L_train_raw_orig, avg_embeddings_train, avg_embeddings_train,
        thresholds = thresholds)
    L_dev_expanded = liger.expand_lfs(
        L_train_raw_orig, L_dev_raw_orig, avg_embeddings_train, avg_embeddings_dev,
        thresholds = thresholds)
    L_test_expanded = liger.expand_lfs(
        L_train_raw_orig, L_test_raw_orig, avg_embeddings_train, avg_embeddings_test,
        thresholds = thresholds)
    
    L_train_raw = L_train_expanded
    L_dev_raw = L_dev_expanded
    L_test_raw = L_test_expanded
    
    T = 1

    L_train = L_train_raw[:L_train_raw.shape[0] - (L_train_raw.shape[0] % T)]
    L_dev = L_dev_raw[:L_dev_raw.shape[0] - (L_dev_raw.shape[0] % T)]
    L_test = L_test_raw[:L_test_raw.shape[0] - (L_test_raw.shape[0] % T)]
    Y_dev = Y_dev_raw[:Y_dev_raw.shape[0] - (Y_dev_raw.shape[0] % T)]
    Y_test = Y_test_raw[:Y_test_raw.shape[0] - (Y_test_raw.shape[0] % T)]

    m_per_task = L_train.shape[1]

    m = T * m_per_task
    v = T
    
    kmeans, embedding_groups, train_cluster_labels = cluster_embeddings(avg_embeddings_train, n_clusters)
    dev_cluster_labels = kmeans.predict(avg_embeddings_dev)
    test_cluster_labels = kmeans.predict(avg_embeddings_test) 
    cluster_models = []
    for i in range(len(embedding_groups)):
        cluster_models.append(Flyingsquid_Cluster(X=embedding_groups[i], mu=kmeans.cluster_centers_[i], T=T, m_per_task=m_per_task))
    
    neg_balances_to_try = np.arange(.01, .99, .05)
    evaluate_thresholds(thresholds, cluster_models, neg_balances_to_try, \
        L_train_expanded, L_dev_expanded, L_test_expanded, \
        Y_dev_raw, Y_test_raw, train_cluster_labels, dev_cluster_labels, test_cluster_labels,\
        evaluate_test=False, tune_test=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config file path')
    args = parser.parse_args()
    main(args)
    
    