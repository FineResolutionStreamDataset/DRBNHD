import subprocess


basins = ['LowerDelaware', 'Neversink', 'LowerWestBranchDelaware', 'Rancocas']
maskpercentages = [0]
random_seeds_default = [42]
model_seeds = [1,2,3]
model_names = ['STGCN','DCRNN','ASTGCN','GWNET', 'STGODE', 'RGCN_v1', 'STID', 'STAEformer', 'STGformer']  # 添加模型列表

for model_name in model_names: 
    for basin in basins:
        for maskpercentage in maskpercentages:
            if maskpercentage == 0:
                random_seeds = [42]
            else:
                random_seeds = random_seeds_default

            for random_seed in random_seeds:
                for model_seed in model_seeds:
                    command = [
                        '...your_path_to//python.exe', 'progress_upscaling.py', ##modify here！！
                        '--basin', str(basin),
                        '--maskpercentage', str(maskpercentage),    
                        '--random_seed', str(random_seed),
                        '--model_seed', str(model_seed),
                        '--model_name', str(model_name)  
                    ]
                    subprocess.run(command)
