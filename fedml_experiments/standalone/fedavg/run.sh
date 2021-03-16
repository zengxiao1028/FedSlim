#!/bin/bash

#python main_fedavg.py --exp_name=standard_6000clients --lr_decay=cosine
#python main_fedavg.py --exp_name=data_IS_only_6000clients --lr_decay=cosine --use_data_IS
#python main_fedavg.py --exp_name=client_IS_only_6000clients --lr_decay=cosine --use_client_IS
#python main_fedavg.py --exp_name=data_client_IS_6000clients --lr_decay=cosine --use_data_IS --use_client_IS

#python main_fedavg.py --exp_name=standard_6000clients --lr_decay=cosine
#python main_fedavg.py --exp_name=data_IS_only_6000clients --lr_decay=cosine --use_data_IS
#python main_fedavg.py --exp_name=data_staleIS_only_6000clients --lr_decay=cosine --use_data_IS --stale_IS_weight
#python main_fedavg.py --exp_name=standard_3000comrounds --lr_decay=cosine --comm_round=3000

#python main_fedavg.py --exp_name=slimmable --lr_decay=cosine --slim_training --wandb_dir=/home/xiao/projects/FedSlim/results

python main_fedavg.py --exp_name=fedslim_adam --lr_decay=cosine --slim_training --slim_width 0.25 0.5 0.75 1.0 --slim_channels=leftmost --server_optimizer=adam --epochs=1