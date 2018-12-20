# -*- coding: utf-8 -*-


"""

factor Analysis
scree plotを実施し、最適な因子数を算出

"""


import pandas as pd
import numpy as np
import argparse
from scipy import stats
from sklearn.cluster import KMeans
from pyclustering.cluster.xmeans import xmeans
import pyclustering.cluster.center_initializer as ci
import pyclustering.utils as utils
from math import pi
import matplotlib as plt
from lib.meshlonlat import *

def main(fldr, filename):
    """
    クラスタリング用のデータを読み込み。
    #全国分足し合わせる
    """

    input_path = "{}/{}".format(fldr, filename)
    fa_path = "{}/{}.factor.csv".format(fldr, filename)
    output_path = "{}/{}.cluster.csv".format(fldr, filename)
    
    df = pd.read_csv(input_path)
    score_df = pd.read_csv(fa_path)
    print(score_df.head(2))
    #_df = df.reindex(columns = ["mesh_CODE"])
    #score_df = pd.merge(_df, score_df, on="mesh_CODE", how="inner")
    
    df["mesh_CODE"] = df["mesh_CODE"].astype(int)
    df = df.set_index("mesh_CODE")
  
    test_array = np.array([score_df["F0"].tolist(),score_df["F1"].tolist(),score_df["F2"].tolist()], np.float).T

    #inshi_with_xmeans = xmeans_clustering(data = score_df, array = test_array)
    inshi_with_xmeans = kmeans_clustering(data = score_df, array = test_array)
    print(inshi_with_xmeans.head(3))
    checkClusterRecords(inshi_with_xmeans)
    
    inshi_with_xmeans["mesh_CODE"] = inshi_with_xmeans["mesh_CODE"].astype(int)
    #inshi_with_xmeans = inshi_with_xmeans.set_index("mesh_CODE")
    df["mesh_CODE"] = df.index
    #df = pd.concat([df, inshi_with_xmeans],  axis=1, join_axes=[df.index])
    df = pd.merge(df, inshi_with_xmeans, on = 'mesh_CODE', how = 'inner')
    df = add_lonlat_from_meshcode(df)
    df.to_csv(output_path, index=False)


def kmeans_clustering(data,array):
	
    kmeans_instance = KMeans(n_clusters = 12, init="k-means++", max_iter=5000).fit(array)
    data["cluster_xmeans"] = 0
    num = data.shape[1]-1
    cluster = kmeans_instance.labels_
    
    #クラスタ番号をデータフレームに付与。
    for n in range(0,len(cluster)):
        data.iat[n,num] = cluster[n]
    
    return data

def xmeans_clustering(data,array):
    """
    create object of X-Means algorithm that uses CCORE for processing
    initial centers - optional parameter, if it is None, then random centers will be used by the algorithm. 
    let's avoid random initial centers and initialize them using K-Means++ method:
    """
    
    initial_centers = ci.kmeans_plusplus_initializer(array, 2).initialize(); 
    xmeans_instance = xmeans(array, initial_centers, kmax = 50, ccore = True);
    
    # run cluster analysis
    # obtain results of clustering

    xmeans_instance.process();    
    clusters = xmeans_instance.get_clusters();
    
    # display allocated clusters 
    #可視化は2次元か3次元のデータしかできないらしい
    #utils.draw_clusters(test2_array, clusters);    
    #後のインデックス指定のため、仮にすべてのクラスタ番号を０にしてカラムを設定。
    data["cluster_xmeans"] = 0
    num = data.shape[1]-1
    
    #クラスタ番号をデータフレームに付与。
    for n in range(0,len(clusters)):
        for i in clusters[n]:
            data.iat[i,num] = n
    
    return data


def checkClusterRecords(inshi_with_xmeans):
    """
    クラスタごとのデータ数を確認
    """

    inshi_with_xmeans["cluster_xmeans"] = inshi_with_xmeans["cluster_xmeans"].astype(int)
    for i in inshi_with_xmeans.groupby("cluster_xmeans"):
      row = i[1]
      print(str(i[0]),"|",len(row))


def add_lonlat_from_meshcode(df):
    """
    メッシュコード列から緯度経度列を付与
    
    """
    meshcode = []
    for index, row in df.iterrows():
        meshcode.append(row.mesh_CODE)
        
    lon = []
    lat = []
    for i in range(len(meshcode)):
        _lat, _lon = mesh2lonlat(str(int(meshcode[i])), 500)
        lon.append(_lon)
        lat.append(_lat)
    df['longitude'] = lon
    df['latitude'] = lat
    return df
    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('-d', '--dir', dest = 'dir', required = True, type = str)
    p.add_argument('-f', '--filename', dest = 'filename', required = True, type = str)
    args = p.parse_args()
    
    main(args.dir, args.filename) 
    