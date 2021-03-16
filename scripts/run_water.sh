#!/bin/bash
cd ../reward_machines
for i in `seq 1 10`; 
do
	for j in `seq 0 1`; 
	do
		# Multi-task
		python3.6 run.py --alg=deepq --env=Water-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql/water/M$i/$j
		python3.6 run.py --alg=deepq --env=Water-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql-rs/water/M$i/$j --use_rs
		python3.6 run.py --alg=deepq --env=Water-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm/water/M$i/$j --use_crm
		python3.6 run.py --alg=deepq --env=Water-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm1/water/M$i/$j --use_crm --num_layers=3 --num_hidden=512
		python3.6 run.py --alg=deepq --env=Water-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm2/water/M$i/$j --use_crm --num_layers=3 --num_hidden=256
		python3.6 run.py --alg=deepq --env=Water-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm3/water/M$i/$j --use_crm --num_layers=6 --num_hidden=64
		python3.6 run.py --alg=deepq --env=Water-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm-rs/water/M$i/$j --use_crm --use_rs
		python3.6 run.py --alg=dhrm --env=Water-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/hrm/water/M$i/$j
		python3.6 run.py --alg=dhrm --env=Water-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/hrm-rs/water/M$i/$j --use_rs

		# Single task
		python3.6 run.py --alg=deepq --env=Water-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql/water-single/M$i/$j
		python3.6 run.py --alg=deepq --env=Water-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql-rs/water-single/M$i/$j --use_rs
		python3.6 run.py --alg=deepq --env=Water-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm/water-single/M$i/$j --use_crm
		python3.6 run.py --alg=deepq --env=Water-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm-rs/water-single/M$i/$j --use_crm --use_rs
		python3.6 run.py --alg=dhrm --env=Water-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/hrm/water-single/M$i/$j
		python3.6 run.py --alg=dhrm --env=Water-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/hrm-rs/water-single/M$i/$j --use_rs
	done
done