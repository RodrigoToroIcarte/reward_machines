#!/bin/bash
cd ../src
for i in `seq 1 10`; 
do
	for j in `seq 0 2`; 
	do
		# Multi-task
		python3.6 run.py --alg=qlearning --env=Craft-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql/craft/M$i/$j
		python3.6 run.py --alg=qlearning --env=Craft-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql-rs/craft/M$i/$j --use_rs
		python3.6 run.py --alg=qlearning --env=Craft-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm/craft/M$i/$j --use_crm
		python3.6 run.py --alg=qlearning --env=Craft-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm-rs/craft/M$i/$j --use_crm --use_rs
		python3.6 run.py --alg=hrm --env=Craft-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/hrm/craft/M$i/$j
		python3.6 run.py --alg=hrm --env=Craft-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/hrm-rs/craft/M$i/$j --use_rs

		# Single task
		python3.6 run.py --alg=qlearning --env=Craft-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql/craft-single/M$i/$j
		python3.6 run.py --alg=qlearning --env=Craft-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/ql-rs/craft-single/M$i/$j --use_rs
		python3.6 run.py --alg=qlearning --env=Craft-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$i/$j --use_crm
		python3.6 run.py --alg=qlearning --env=Craft-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm-rs/craft-single/M$i/$j --use_crm --use_rs
		python3.6 run.py --alg=hrm --env=Craft-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/hrm/craft-single/M$i/$j
		python3.6 run.py --alg=hrm --env=Craft-single-M$i-v0 --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/hrm-rs/craft-single/M$i/$j --use_rs
	done
done