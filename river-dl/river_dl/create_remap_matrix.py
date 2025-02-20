import numpy as np
import pandas as pd

def remap_matrix(data, data_NHD, crosswalk_path):
    crossDF = pd.read_csv(crosswalk_path)
    idx_segs = data_NHD['segs']
    col_segs = data['segs']
    col_segs = np.array(col_segs, dtype=np.int64)  # Convert to numpy.int64

    missing_comids = [comid for comid in idx_segs if comid not in crossDF['COMID'].values]
    if missing_comids:
        raise ValueError(f'The following COMIDs are missing in the crosswalk: {missing_comids}')

    crossDF = crossDF.loc[crossDF['COMID'].isin(idx_segs)]
    # Fill missing values in 'order_up_to_down' and keep the maximum value for each 'seg_id_nat'
    crossDF['order_up_to_down'].fillna(-1, inplace=True)

    # Initialize the adjacency matrix
    adj_cropped = np.zeros((len(idx_segs), len(col_segs)))
    # Mapping for COMID and seg_id_nat to matrix indices
    comid_to_index = {comid: idx for idx, comid in enumerate(idx_segs)}
    segid_to_index = {seg_id: idx for idx, seg_id in enumerate(col_segs)}

    for comid in idx_segs:
        matching_row = crossDF[crossDF['COMID'] == comid]
        if not matching_row.empty:
            seg_id_nat = matching_row.iloc[0]['seg_id_nat']
            comid_idx = comid_to_index.get(comid)
            segid_idx = segid_to_index.get(seg_id_nat)
            if comid_idx is not None and segid_idx is not None:
                adj_cropped[comid_idx, segid_idx] = 1  # Setting relationship
        else:
            raise ValueError(f'COMID {comid} does not match any entry in the crosswalk')
    return adj_cropped

# Example usage
if __name__ == "__main__":
    data = np.load('../workflow_examples/output_downscale_NHM/prepped.npz', allow_pickle=True)
    data_NHD = np.load('../workflow_examples/output_downscale_NHM/prepped_NHD.npz', allow_pickle=True)
    crosswalk_path = '../DRB_NHD_on_NHM_20240119/NHD_on_NHM_crosswalk.csv'
    a = remap_matrix(data, data_NHD, crosswalk_path)
    print(a)
    print(a.shape)  # Ensure the shape is (63, 13)
