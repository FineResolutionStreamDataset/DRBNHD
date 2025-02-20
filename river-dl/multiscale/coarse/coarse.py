import os
import sys
import yaml
import xarray as xr
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, '../config_multiscale.yml')
with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../..')))


import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import importlib
import sys
code_dir = config['code_dir']
sys.path.insert(1, code_dir)
outdir = config['out_dir']

from multiscale.coarse.predict import predict_from_io_data
from multiscale.coarse.torch_utils import train_torch
from river_dl.mypreproc_utils import prep_all_data
from river_dl.mypreproc_utils import reduce_training_data_random
from river_dl.torch_utils import rmse_masked as rmse_masked
from river_dl.evaluate import combined_metrics
from pandas import pandas as pd




import argparse
parser = argparse.ArgumentParser(description='Running multi-parameter experiments')
parser.add_argument('--basin', type=str, help='basin', default='Rancocas')
parser.add_argument('--maskpercentage', type=float, help='maskpercentage', default=0)
parser.add_argument('--random_seed', type=int, help='random_seed', default=42)
parser.add_argument('--model_seed', type=int, help='model_seed', default=1)
parser.add_argument('--model_name', type=str, help='model_name',default='STGCN')
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





def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)



def getSegs(x):
    #x = 'COMID' or 'seg_id_nat'
    crossDF = pd.read_csv("../../DRB_NHD_on_NHM_20240119/NHD_on_NHM_crosswalk.csv")
    crossDF = crossDF.loc[crossDF.Basin==basin]
    return np.unique(crossDF[x].values)





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
          #segs=getSegs('seg_id_nat'),
          out_file=f"../{outdir}/prepped.npz",
          trn_offset=config['trn_offset'],
          tst_val_offset=config['tst_val_offset'],
          check_pre_partitions=False,
          fill_batch=False)





#fine-tune
data = np.load(f"../{outdir}/prepped.npz")
num_segs = len(np.unique(data['ids_trn']))
adj_mx = data['dist_matrix']
in_dim = len(data['x_vars'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model(input_dim=in_dim, hidden_dim=config['hidden_size'], adj_matrix=adj_mx, device=device, seed=model_seed)
opt = optim.Adam(model.parameters(), lr=config['finetune_learning_rate'])
if 'pre_train' in locals():
    model.load_state_dict(torch.load(f"../{outdir}/pretrained_weights.pth", map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))



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
            weights_file=f"../{outdir}/finetuned_weights.pth",
            log_file=f"../{outdir}/base_log.csv",
            device=device)


#predict
data = np.load(f"../{outdir}/prepped.npz")
in_dim = len(data['x_vars'])
adj_mx = data['dist_matrix']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model(in_dim, config['hidden_size'], adj_matrix=adj_mx, device=device, seed=model_seed)
model.load_state_dict(torch.load(f"../{outdir}/finetuned_weights.pth"))
partitions = ['trn','val','tst']


for partition in partitions:
    outfile = f"your_path_to_river-dl/results/results_coarse/{MODEL_NAME}/preds/{basin}_{maskpercentage_formatted}_{partition}_{model_seed}_preds.feather"
    ensure_dir(outfile)
    
    predict_from_io_data(model=model,
                        io_data=f"../{outdir}/prepped.npz",
                        partition=partition,
                        outfile=outfile,
                        log_vars=False,
                        trn_offset=config['trn_offset'],
                        tst_val_offset=config['tst_val_offset'],
                        spatial_idx_name="seg_id_nat")

    # 检查预测结果文件
    pred_df = pd.read_feather(outfile)






def get_grp_arg(metric_type):
    if metric_type == 'overall':
        return None
    elif metric_type == 'month':
        return 'month'
    elif metric_type == 'reach':
        return 'seg_id_nat'
    elif metric_type == 'month_reach':
        return ['seg_id_nat', 'month']

metric_types = ['overall', 'month', 'reach', 'month_reach']



for metric_type in metric_types:
    grp_arg = get_grp_arg(metric_type)
    outfile = f"your_path_to_river-dl/results/results_coarse/{MODEL_NAME}/metrics/{metric_type}/{basin}_{maskpercentage_formatted}_{model_seed}_metrics.csv"
    ensure_dir(outfile)
    
    combined_metrics(obs_file=config['obs_file'],
                    #pred_trn=f"../{outdir}/trn_preds.feather",
                    #pred_val=f"../{outdir}/val_preds.feather",
                    #pred_tst=f"../{outdir}/tst_preds.feather",
                    pred_trn=f"your_path_to_river-dl/results/results_coarse/{MODEL_NAME}/preds/{basin}_{maskpercentage_formatted}_trn_{model_seed}_preds.feather",
                    pred_val=f"your_path_to_river-dl/results/results_coarse/{MODEL_NAME}/preds/{basin}_{maskpercentage_formatted}_val_{model_seed}_preds.feather",
                    pred_tst=f"your_path_to_river-dl/results/results_coarse/{MODEL_NAME}/preds/{basin}_{maskpercentage_formatted}_tst_{model_seed}_preds.feather",
                    group_spatially=False if not grp_arg else True if "COMID" in grp_arg else False,
                    group_temporally=False if not grp_arg else 'M' if "month" in grp_arg else False,
                    outfile=outfile,
                    spatial_idx_name="seg_id_nat")




