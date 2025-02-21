import os
import sys
import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, '../config_multiscale.yml')
with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../..')))



import numpy as np
import torch
import torch.optim as optim
import importlib
import sys
code_dir = config['code_dir']
sys.path.insert(1, code_dir)
outdir = config['out_dir']





from multiscale.progress_upscaling.predict_remap import mypredict_from_io_data_remap
from multiscale.progress_upscaling.predict import predict_from_io_data
from multiscale.progress_upscaling.torch_utils_remap import (train_torch)

from river_dl.mypreproc_utils import prep_all_data
from river_dl.my_torch_utilsMulti import rmse_masked
from river_dl.evaluate import combined_metrics
from river_dl.mypreproc_utils import reduce_training_data_random
from river_dl.create_mixed_distance_matrix import mixed_dixtance_matrix
from river_dl.create_remap_matrix import remap_matrix
from pandas import pandas as pd
import xarray as xr

import argparse
parser = argparse.ArgumentParser(description='Running multi-parameter experiments')
parser.add_argument('--basin', type=str, help='basin', default='Rancocas')
parser.add_argument('--maskpercentage', type=float, help='maskpercentage', default=0)
parser.add_argument('--random_seed', type=int, help='random_seed', default=42)
parser.add_argument('--model_seed', type=int, help='model_seed', default=1)
parser.add_argument('--model_name', type=str, help='model_name',default='RGCN_v1')
args = parser.parse_args()


basin = args.basin
maskpercentage = args.maskpercentage
random_seed = args.random_seed
model_seed = args.model_seed
if maskpercentage > 0.99:       
    maskpercentage_formatted = int(maskpercentage * 1000)
else:
    maskpercentage_formatted = int(maskpercentage * 100)    
MODEL_NAME = args.model_name




model_module = importlib.import_module(f"multiscale.MODEL.{MODEL_NAME}")
Model = getattr(model_module, MODEL_NAME)


def getSegs(x):
    #x = 'COMID' or 'seg_id_nat'
    crossDF = pd.read_csv("../../DRB_NHD_on_NHM_20240119/NHD_on_NHM_crosswalk.csv")
    crossDF = crossDF.loc[crossDF.Basin==basin]
    return np.unique(crossDF[x].values)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)



if maskpercentage > 0:
    y_train_sparse = reduce_training_data_random(
        config['obs_file_NHD'],
        train_start_date=config['train_start_date'],
        train_end_date=config['train_end_date'],
        val_start_date=config['val_start_date'],
        val_end_date=config['val_end_date'],
        reduce_amount=maskpercentage,
        out_file=f"../{outdir}/spare_obs_temp_flow",
        #segs=None
        segs=getSegs('COMID'),
        random_seed=random_seed,
    )

#Prepare DRB


prepped = prep_all_data(
          x_data_file=config['sntemp_file'],
          pretrain_file=config['sntemp_file'],
          y_data_file=config['obs_file'],
          distfile=config['dist_matrix_file'],
          x_vars=config['x_vars'],
          y_vars_pretrain=config['y_vars_pretrain'],
          y_vars_finetune=config['y_vars_finetune'],
          spatial_idx_name="seg_id_nat",
          catch_prop_file=None,
          train_start_date=config['train_start_date'],
          train_end_date=config['train_end_date'],
          val_start_date=config['val_start_date'],
          val_end_date=config['val_end_date'],
          test_start_date=config['test_start_date'],
          test_end_date=config['test_end_date'],
          segs=None,
          #segs = getSegs('seg_id_nat'),
          out_file=f"../{outdir}/prepped.npz",
          trn_offset=config['trn_offset'],
          tst_val_offset=config['tst_val_offset'],
          check_pre_partitions=False,
          fill_batch=False)


prepped_NHD =     prep_all_data(
                  x_data_file=config['sntemp_file_NHD'],
                  pretrain_file=config['sntemp_file_NHD'],
                  y_data_file=f"../{outdir}/spare_obs_temp_flow" if maskpercentage > 0 else config['obs_file_NHD'],
                  distfile=config['dist_matrix_file_NHD'],
                  x_vars=config['x_vars_NHD'],
                  y_vars_pretrain=config['y_vars_pretrain_NHD'],
                  y_vars_finetune=config['y_vars_finetune'],
                  spatial_idx_name="COMID",
                  catch_prop_file=None,
                  train_start_date=config['train_start_date'],
                  train_end_date=config['train_end_date'],
                  val_start_date=config['val_start_date'],
                  val_end_date=config['val_end_date'],
                  test_start_date=config['test_start_date'],
                  test_end_date=config['test_end_date'],
                  #segs=None,
                  segs=getSegs('COMID'),
                  out_file=f"../{outdir}/prepped_NHD.npz",
                  trn_offset=config['trn_offset'],
                  tst_val_offset=config['tst_val_offset'],
                  check_pre_partitions=False,
                  fill_batch=False)

#Train DRB_RGrN
data = np.load(f"../{outdir}/prepped.npz")
data_NHD = np.load(f"../{outdir}/prepped_NHD.npz")
num_segs = len(np.unique(data['ids_trn']))
adj_mx = data['dist_matrix']
in_dim = len(data['x_vars'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
remap_mx = remap_matrix(data,data_NHD,config['crosswalk'])
model = Model(input_dim=in_dim, hidden_dim=config['hidden_size'], adj_matrix=adj_mx, remap_matrix=remap_mx, device=device, seed=model_seed)

opt = optim.Adam(model.parameters(), lr=config['finetune_learning_rate'])
train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_trn'],
            y_train=data['y_obs_trn'],
            x_val=data['x_val'],
            y_val=data['y_obs_val'],
            max_epochs=40,
            early_stopping_patience=config['early_stopping'],
            batch_size=num_segs,
            weights_file=f"../{outdir}/DRB_weights.pth",
            log_file=f"../{outdir}/finetune_log.csv",
            device=device)





#Predict DRB_NHD
data = np.load(f"../{outdir}/prepped.npz")
data_NHD = np.load(f"../{outdir}/prepped_NHD.npz")
x_num_segs = len(np.unique(data['ids_trn']))
x_num_segs_NHD = len(np.unique(data_NHD['ids_trn']))
adj_mx = data['dist_matrix']
adj_mx_NHD = data_NHD['dist_matrix']
adj_mx_mixed = mixed_dixtance_matrix(data,data_NHD, config['dist_matrix_file_NHD'], config['crosswalk'])
remap_mx = remap_matrix(data,data_NHD,config['crosswalk'])
in_dim = len(data['x_vars'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model(input_dim=in_dim, hidden_dim=config['hidden_size'], adj_matrix=adj_mx, remap_matrix=remap_mx, device=device, seed=model_seed)
model.load_state_dict(torch.load(f"../{outdir}/DRB_weights.pth", map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
partitions = ['trn','val','tst']
for partition in partitions:
    mypredict_from_io_data_remap(model=model,
                         io_data_x=f"../{outdir}/prepped.npz",
                         io_data_x2=f"../{outdir}/prepped_NHD.npz",
                         io_data_y=f"../{outdir}/prepped.npz",
                         partition=partition,
                         outfile=f"../{outdir}/{partition}_preds_DRB.feather",
                         log_vars=False,
                         trn_offset=config['trn_offset'],
                         tst_val_offset=config['tst_val_offset'],
                         spatial_idx_name="COMID")


def filter_preds_by_segs(preds_file, segs, output_file):
    predsDF = pd.read_feather(preds_file)
    filteredDF = predsDF[predsDF['COMID'].isin(segs)]
    filteredDF.reset_index(drop=True, inplace=True) # 重置索引以使其成为默认索引
    filteredDF.to_feather(output_file)
    return filteredDF
#segs = getSegs('seg_id_nat')
segs = getSegs('COMID')
filter_preds_by_segs(f"../{outdir}/trn_preds_DRB.feather", segs, f"../{outdir}/trn_preds_NHM.feather")
filter_preds_by_segs(f"../{outdir}/val_preds_DRB.feather", segs, f"../{outdir}/val_preds_NHM.feather")
filter_preds_by_segs(f"../{outdir}/tst_preds_DRB.feather", segs, f"../{outdir}/tst_preds_NHM.feather")


ds = xr.open_zarr(f"../{outdir}/spare_obs_temp_flow" if maskpercentage > 0 else config['obs_file_NHD'])
df = ds.to_dataframe().reset_index()
segs = getSegs('COMID')
df0 = df[df['COMID'].isin(segs)]
df0_non_nan = df0.dropna(subset=['mean_temp_c'])

a1 = pd.read_feather(f"../{outdir}/trn_preds_DRB.feather")
a2 = pd.read_feather(f"../{outdir}/val_preds_DRB.feather")
a3 = pd.read_feather(f"../{outdir}/tst_preds_DRB.feather")
a = pd.concat([a1, a2, a3], ignore_index=True)


merged_df = df.merge(a, on=['date', 'COMID'], how='left', suffixes=('', '_pred'))

def update_temp(row):
    return row['mean_temp_c_pred']

merged_df['mean_temp_c'] = merged_df.apply(update_temp, axis=1)
merged_df = merged_df.drop(columns=['mean_temp_c_pred'])
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df = merged_df.set_index(['date', 'COMID'])
remap_ds = merged_df.to_xarray()
remap_ds = remap_ds.assign_coords(**ds.coords)
remap_ds.attrs = ds.attrs
remap_ds.to_zarr(f"../{outdir}/remap_obs_temp_flow", mode='w')




prepped_NHD = prep_all_data(
              x_data_file=config['sntemp_file_NHD'],
              pretrain_file=config['sntemp_file_NHD'],
              y_data_file=f"../{outdir}/remap_obs_temp_flow", ##remap_obs
              distfile=config['dist_matrix_file_NHD'],
              x_vars=config['x_vars_NHD'],
              y_vars_pretrain=config['y_vars_pretrain_NHD'],
              y_vars_finetune=config['y_vars_finetune'],
              spatial_idx_name='COMID',
              catch_prop_file=None,
              train_start_date=config['train_start_date'],
              train_end_date=config['train_end_date'],
              val_start_date=config['val_start_date'],
              val_end_date=config['val_end_date'],
              test_start_date=config['test_start_date'],
              test_end_date=config['test_end_date'],
              #segs=None,
              segs=getSegs('COMID'),
              out_file=f"../{outdir}/prepped_NHD_long.npz",
              trn_offset=config['trn_offset'],
              tst_val_offset=config['tst_val_offset'],
              check_pre_partitions=False,
              fill_batch=False)





#Pretrain MSGL by simulation (prepped_long.npz,prepped_NHD_long.npz)
data = np.load(f"../{outdir}/prepped_NHD_long.npz")
num_segs = len(np.unique(data['ids_trn']))
adj_mx = data['dist_matrix']
in_dim = len(data['x_vars'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model(input_dim=in_dim, hidden_dim=config['hidden_size'], adj_matrix=adj_mx, device=device, seed=model_seed)
opt = optim.Adam(model.parameters(),lr=config['finetune_learning_rate'])
train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_trn'],
            y_train=data['y_obs_trn'],
            x_val=data['x_val'],
            y_val=data['y_obs_val'],
            x_tst=data['x_tst'],
            y_tst=data['y_obs_tst'],
            max_epochs=config['pt_epochs'],
            early_stopping_patience=config['early_stopping'],
            batch_size=num_segs,
            weights_file=f"../{outdir}/{basin}_pretrained_weights_Multi.pth",
            log_file=f"../{outdir}/700pretrain_log.csv",
            device=device)





prepped_NHD =     prep_all_data(
                  x_data_file=config['sntemp_file_NHD'],
                  pretrain_file=config['sntemp_file_NHD'],
                  y_data_file=f"../{outdir}/spare_obs_temp_flow" if maskpercentage > 0 else config['obs_file_NHD'],
                  distfile=config['dist_matrix_file_NHD'],
                  x_vars=config['x_vars_NHD'],
                  y_vars_pretrain=config['y_vars_pretrain_NHD'],
                  y_vars_finetune=config['y_vars_finetune'],
                  spatial_idx_name='COMID',
                  catch_prop_file=None,
                  train_start_date=config['train_start_date'],
                  train_end_date=config['train_end_date'],
                  val_start_date=config['val_start_date'],
                  val_end_date=config['val_end_date'],
                  test_start_date=config['test_start_date'],
                  test_end_date=config['test_end_date'],
                  #segs=None,
                  segs=getSegs('COMID'),
                  out_file=f"../{outdir}/prepped_NHD.npz",
                  trn_offset=config['trn_offset'],
                  tst_val_offset=config['tst_val_offset'],
                  check_pre_partitions=False)


#Finetune MSGL by True Label(prepped.npz, prepped_NHD.npz)
data = np.load(f"../{outdir}/prepped_NHD.npz")
num_segs = len(np.unique(data['ids_trn']))
adj_mx = data['dist_matrix']
in_dim = len(data['x_vars'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model(input_dim=in_dim, hidden_dim=config['hidden_size'], adj_matrix=adj_mx, device=device, seed=model_seed)
model.load_state_dict(torch.load(f"../{outdir}/{basin}_pretrained_weights_Multi.pth", map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
print(f"Running on device: {device}")
opt = optim.Adam(model.parameters(),lr=config['finetune_learning_rate'])
train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_trn'],
            y_train=data['y_obs_trn'],
            x_val=data['x_val'],
            y_val=data['y_obs_val'],
            x_tst=data['x_tst'],
            y_tst=data['y_obs_tst'],
            max_epochs=config['ft_epochs'],
            early_stopping_patience=config['early_stopping'],
            batch_size=num_segs,
            weights_file=f"../{outdir}/finetuned_weights_Multi.pth",
            log_file=f"../{outdir}/700finetune_log.csv",
            device=device)




data = np.load(f"../{outdir}/prepped_NHD.npz")
adj_mx = data['dist_matrix']
in_dim = len(data['x_vars'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model(input_dim=in_dim, hidden_dim=config['hidden_size'], adj_matrix=adj_mx, device=device, seed=model_seed)
model.load_state_dict(torch.load(f"../{outdir}/finetuned_weights_Multi.pth", map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
partitions = ['trn','val','tst']

for partition in partitions:
    outfile = f"your path to river-dl/results/results_progress_upscaling/{MODEL_NAME}/preds/{basin}_{maskpercentage_formatted}_{partition}_{model_seed}_preds.feather"
    ensure_dir(outfile)
    predict_from_io_data(model=model,
                         io_data=f"../{outdir}/prepped_NHD.npz",
                         partition=partition,
                         outfile=outfile,
                         log_vars=False,
                         trn_offset=config['trn_offset'],
                         tst_val_offset=config['tst_val_offset'],
                         spatial_idx_name="COMID")

def get_grp_arg(metric_type):
    if metric_type == 'overall':
        return None
    elif metric_type == 'month':
        return 'month'
    elif metric_type == 'reach':
        return 'COMID'
    elif metric_type == 'month_reach':
        return ['COMID', 'month']
metric_types = ['overall', 'month', 'reach', 'month_reach']


for metric_type in metric_types:
    grp_arg = get_grp_arg(metric_type)
    outfile = f"your path to river-dl/results/results_progress_upscaling/{MODEL_NAME}/metrics/{metric_type}/{basin}_{maskpercentage_formatted}_{model_seed}_metrics.csv"
    ensure_dir(outfile)
    combined_metrics(obs_file=config['obs_file_NHD'],
                     #pred_trn=f"../{outdir}/trn_preds.feather",
                     #pred_val=f"../{outdir}/val_preds.feather",
                     #pred_tst=f"../{outdir}/tst_preds.feather",
                     pred_trn=f"your path to river-dl/results/results_progress_upscaling/{MODEL_NAME}/preds/{basin}_{maskpercentage_formatted}_trn_{model_seed}_preds.feather",
                     pred_val=f"your path to river-dl/results/results_progress_upscaling/{MODEL_NAME}/preds/{basin}_{maskpercentage_formatted}_val_{model_seed}_preds.feather",
                     pred_tst=f"your path to river-dl/results/results_progress_upscaling/{MODEL_NAME}/preds/{basin}_{maskpercentage_formatted}_tst_{model_seed}_preds.feather",
                     group_spatially=False if not grp_arg else True if "COMID" in grp_arg else False,
                     group_temporally=False if not grp_arg else 'M' if "month" in grp_arg else False,
                     outfile=outfile,
                     spatial_idx_name="COMID")







