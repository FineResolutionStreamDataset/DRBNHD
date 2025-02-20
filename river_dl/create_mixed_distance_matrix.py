# -*- coding: utf-8 -*-



import os
import numpy as np
import pandas as pd


def mixed_dixtance_matrix(data,data_NHD,distance_matrix,crosswalk):
    distance_matrix_DRB_NHD = np.load(distance_matrix) #4.17
    crossDF = pd.read_csv(crosswalk) #4.17

    idx_segs = data_NHD['segs']
    col_segs = data['segs']
    col_segs = np.array(col_segs, dtype=np.int64)  # Convert to numpy.int64

    distanceDF = pd.DataFrame(distance_matrix_DRB_NHD['updown'],index=distance_matrix_DRB_NHD["rowcolnames"],columns=distance_matrix_DRB_NHD["rowcolnames"])
    distanceDF = distanceDF.loc[idx_segs, idx_segs] #Crop DF from (3221,3221) to (63,63)
    crossDF = crossDF.loc[crossDF['COMID'].isin(idx_segs)]


    crossDF['order_up_to_down'].fillna(-1, inplace=True)#4.18 从小到大
    crossDF = crossDF.loc[crossDF.groupby('seg_id_nat')['order_up_to_down'].transform(max)==crossDF['order_up_to_down']]




    distanceNHD_NHM = distanceDF[crossDF.COMID]
    distanceNHD_NHM.columns = [crossDF.loc[crossDF.COMID==x,"seg_id_nat"].values[0] for x in distanceNHD_NHM.columns]
    adj = distanceNHD_NHM.to_numpy()
    adj = np.where(np.isinf(adj), 0, adj)
    adj = -adj
    mean_adj = np.mean(adj[adj != 0])
    std_adj = np.std(adj[adj != 0])
    adj[adj != 0] = adj[adj != 0] - mean_adj
    adj[adj != 0] = adj[adj != 0] / std_adj
    adj[adj != 0] = 1 / (1 + np.exp(-adj[adj != 0]))

    #I = np.eye(adj.shape[0])
    #A_hat = adj.copy() + I
    A_hat = adj
    D = np.sum(A_hat, axis=1)
    D_inv = D ** -1.0
    D_inv = np.diag(D_inv)
    A_hat = np.matmul(D_inv, A_hat)

    return A_hat
    #distance_matrix_mixed = {'updown':A_hat,'rownames':idx_segs,'colnames':col_segs}
    #np.savez_compressed(os.path.join(sharepointPath,"dist_matrix_mixed.npz"), **distance_matrix_mixed)



if __name__ == "__main__":
    data = np.load('../workflow_examples/output_downscale_NHM/prepped.npz')
    data_NHD = np.load('../workflow_examples/output_downscale_NHM/prepped_NHD.npz')
    distance_matrix_path = '../DRB_NHD_on_NHM_20240119/dist_matrix.npz'
    crosswalk_path = '../DRB_NHD_on_NHM_20240119/NHD_on_NHM_crosswalk.csv'
    adj_mx_mixed = mixed_dixtance_matrix(data, data_NHD, distance_matrix_path, crosswalk_path)


