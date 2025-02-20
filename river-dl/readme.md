The DRB_NHD_on_NHM_all_2024 and DRB_NHM_20230928 at github are not complete.

Download the DRB_NHD_on_NHM_all_2024 at anonymous Google Drive:
https://drive.google.com/file/d/1KS507tSReqiQ3RSxvNUCNOiH2oBduxa1/view?usp=drive_link

Download the DRB_NHM_20230928 at anonymous Google Drive:
https://drive.google.com/file/d/18EBbCC0x7jJfkxuLGIHTOHNjr7WoSfrj/view?usp=drive_link


To train the SpatioTemporal model by low-resolution DRB dataset, run coarse.py
To train the SpatioTemporal model by high-resolution DRB dataset, run base.py
To use the basic downscaling method, run remap.py, direct_upscaling.py, progress_upscaling.py

All of the spatiotemporal models are in the multiscale.MODEL with reference. All of the models are kept the same with
their original github verison, but the code is slight for the first layer and final layer of the forward function. 

The input of DRB Stream temperature data is [batch_size(number of nodes), sequence_length(number of days), number of features(7)], 

The output of DRB Stream temperature data is [batch_size(number of nodes), sequence_length(number of days), number of features(1)].
