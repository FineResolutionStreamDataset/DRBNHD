import pandas as pd
import numpy as np
import xarray as xr
import datetime
import torch
from numpy.lib.npyio import NpzFile
from river_dl.postproc_utils import prepped_array_to_df




def get_data_if_file(d):
    """
    rudimentary check if data .npz file is already loaded. if not, load it
    :param d:
    :return:
    """
    if isinstance(d, NpzFile) or isinstance(d, dict):
        return d
    else:
        return np.load(d, allow_pickle=True)


def unscale_output(y_scl, y_std, y_mean, y_vars, log_vars=None):
    """
    unscale output data given a standard deviation and a mean value for the
    outputs
    :param y_scl: [pd dataframe] scaled output data (predicted or observed)
    :param y_std:[numpy array] array of standard deviation of variables_to_log [n_out]
    :param y_mean:[numpy array] array of variable means [n_out]
    :param y_vars: [list-like] y_dataset variable names
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    prep
    :return: unscaled data
    """
    y_unscaled = y_scl.copy()
    # I'm replacing just the variable columns. I have to specify because, at
    # least in some cases, there are other columns (e.g., "seg_id_nat" and
    # date")
    y_unscaled[y_vars] = (y_scl[y_vars] * y_std) + y_mean
    if log_vars:
        y_unscaled[log_vars] = np.exp(y_unscaled[log_vars])
    return y_unscaled


def mypredict_from_io_data(
        model,
        io_data_x,
        io_data_x2,
        io_data_y,
        partition,
        outfile,
        log_vars=False,
        trn_offset=1.0,
        tst_val_offset=1.0,
        spatial_idx_name="seg_id_nat",
        time_idx_name="date",
        trn_latest_time=None,
        val_latest_time=None,
        tst_latest_time=None
):
    """
    make predictions from trained model
    :param io_data: [str] directory to prepped data file
    :param partition: [str] must be 'trn' or 'tst'; whether you want to predict
    for the train or the dev period
    :param outfile: [str] the file where the output data should be stored
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    prep
    :param trn_offset: [str] value for the training offset
    :param tst_val_offset: [str] value for the testing and validation offset
    :param trn_latest_time: [str] when specified, the training partition preds will
    be trimmed to use trn_latest_time as the last date
    :param val_latest_time: [str] when specified, the validation partition preds will
    be trimmed to use val_latest_time as the last date
    :param tst_latest_time: [str] when specified, the test partition preds will
    be trimmed to use tst_latest_time as the last date
    :return: [pd dataframe] predictions
    """
    io_data_x = get_data_if_file(io_data_x)
    io_data_x2 = get_data_if_file(io_data_x2)
    io_data_y = get_data_if_file(io_data_y)

    if partition == "trn":
        keep_portion = trn_offset
        if trn_latest_time:
            latest_time = trn_latest_time
        else:
            latest_time = None
    elif partition == "val":
        keep_portion = tst_val_offset
        if val_latest_time:
            latest_time = val_latest_time
        else:
            latest_time = None
    elif partition == "tst":
        keep_portion = tst_val_offset
        if tst_latest_time:
            latest_time = tst_latest_time
        else:
            latest_time = None

    preds = predict(
        model,
        io_data_x[f"x_{partition}"],
        io_data_x2[f"x_{partition}"],
        io_data_x[f"ids_{partition}"],
        io_data_x2[f"ids_{partition}"],
        io_data_x2[f"times_{partition}"],
        io_data_y["y_std"],
        io_data_y["y_mean"],
        io_data_y["y_obs_vars"],
        keep_last_portion=keep_portion,
        outfile=outfile,
        log_vars=log_vars,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name,
        latest_time=latest_time,
        pad_mask=io_data_y[f"padded_{partition}"]
    )
    return preds


def predict(
        model,
        x_data,
        x_data2,
        pred_ids,
        pred_ids2,
        pred_dates,
        y_stds,
        y_means,
        y_vars,
        keep_last_portion=1,
        outfile=None,
        log_vars=False,
        spatial_idx_name="seg_id_nat",
        time_idx_name="date",
        latest_time=None,
        pad_mask=None,
):
    """
    use trained model to make predictions
    :param model: [tf model] trained TF model to use for predictions
    :param x_data: [np array] numpy array of scaled and centered x_data
    :param pred_ids: [np array] the ids of the segments (same shape as x_data)
    :param pred_dates: [np array] the dates of the segments (same shape as
    x_data)
    :param keep_last_portion: [float] fraction of the predictions to keep starting
    from the *end* of the predictions (0-1). (1 means you keep all of the
    predictions, .75 means you keep the final three quarters of the predictions). Alternatively, if
    keep_last_portion is > 1 it's taken as an absolute number of predictions to retain from the end of the
    prediction sequence.
    :param y_stds:[np array] the standard deviation of the y_dataset data
    :param y_means:[np array] the means of the y_dataset data
    :param y_vars:[np array] the variable names of the y_dataset data
    :param outfile: [str] the file where the output data should be stored
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    :param latest_time: [str] when provided, the latest time that should be included
    in the returned dataframe
    :param pad_mask: [np array] bool array with True for padded data and False
    otherwise
    :return: out predictions
    """

    num_segs = len(np.unique(pred_ids))
    num_segs2 = len(np.unique(pred_ids2))

    if issubclass(type(model), torch.nn.Module):
        y_pred = predict_torch(x_data,x_data2, model, batch_size=num_segs,batch_size2 = num_segs2)
    else:
        raise TypeError("Model must be a torch.nn.Module or tf.Keras.Model")

    if keep_last_portion > 1:
        frac_seq_len = int(keep_last_portion)
    else:
        frac_seq_len = round(pred_ids2.shape[1] * (keep_last_portion))

    y_pred = y_pred[:, -frac_seq_len:, ...]

    # set to nan the data that were added to fill batches
    if pad_mask is not None:
        pad_mask = torch.tensor(pad_mask[:, -frac_seq_len:, ...])
        y_pred[pad_mask] = np.nan
    pred_ids2 = pred_ids2[:, -frac_seq_len:, ...]
    pred_dates = pred_dates[:, -frac_seq_len:, ...]


    np.save("y_pred.npy",y_pred) #2023

    y_pred_pp = prepped_array_to_df(y_pred, pred_dates, pred_ids2, y_vars, spatial_idx_name, time_idx_name)#
    y_pred_pp = unscale_output(y_pred_pp, y_stds, y_means, y_vars, log_vars)



    # remove data that were added to fill batches
    y_pred_pp.dropna(inplace=True)
    y_pred_pp = y_pred_pp.reset_index().drop(columns='index')

    # Cut off the end times if specified
    if latest_time:
        y_pred_pp = (y_pred_pp.drop(y_pred_pp[y_pred_pp[time_idx_name] > np.datetime64(latest_time)].index)
                     .reset_index()
                     .drop(columns='index')
                     )

    if outfile:
        y_pred_pp.to_feather(outfile)
    return y_pred_pp


def predict_torch(x_data,x_data2, model, batch_size, batch_size2):
    """
    @param model: [object] initialized torch model
    @param batch_size: [int]
    @param device: [str] cuda or cpu
    @return: [tensor] predicted values
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = []
    data2 = []
    for i in range(len(x_data)):
        data.append(torch.from_numpy(x_data[i]).float())
    for i in range(len(x_data2)):
        data2.append(torch.from_numpy(x_data2[i]).float())
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
    dataloader2 = torch.utils.data.DataLoader(data2, batch_size=batch_size2,shuffle=False, pin_memory=True)
    model.eval()
    predicted = []
    for (x, x2) in zip(dataloader, dataloader2):
        trainx = x.to(device)
        trainx2 = x2.to(device)
        with torch.no_grad():
            _,hidden_seq = model.forward_task1(trainx)
            output = model.forward_task2(trainx2,hidden_seq).detach().cpu()
        predicted.append(output)
    predicted = torch.cat(predicted, dim=0)
    return predicted










