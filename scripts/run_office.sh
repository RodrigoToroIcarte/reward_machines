#!/bin/bash
cd ../reward_machines
for i in `seq 0 59`; 
do
	# Multi-task
	python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql/office/M1/$i
	python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql-rs/office/M1/$i --use_rs
	python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/crm/office/M1/$i --use_crm
	python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/crm-rs/office/M1/$i --use_crm --use_rs
	python3.6 run.py --alg=hrm --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/hrm/office/M1/$i
	python3.6 run.py --alg=hrm --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/hrm-rs/office/M1/$i --use_rs

	# Single task
	python3.6 run.py --alg=qlearning --env=Office-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql/office-single/M1/$i
	python3.6 run.py --alg=qlearning --env=Office-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql-rs/office-single/M1/$i --use_rs
	python3.6 run.py --alg=qlearning --env=Office-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/crm/office-single/M1/$i --use_crm
	python3.6 run.py --alg=qlearning --env=Office-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/crm-rs/office-single/M1/$i --use_crm --use_rs
	python3.6 run.py --alg=hrm --env=Office-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/hrm/office-single/M1/$i
	python3.6 run.py --alg=hrm --env=Office-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/hrm-rs/office-single/M1/$i --use_rs
done