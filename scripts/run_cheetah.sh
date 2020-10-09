#!/bin/bash
cd ../src
for i in `seq 0 9`; 
do
	# Task 1
	python3.6 run.py --alg=ddpg --env=Half-Cheetah-RM1-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/ql/cheetah/M1/$i
	python3.6 run.py --alg=ddpg --env=Half-Cheetah-RM1-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/ql-rs/cheetah/M1/$i --use_rs
	python3.6 run.py --alg=ddpg --env=Half-Cheetah-RM1-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/crm/cheetah/M1/$i --use_crm
	python3.6 run.py --alg=ddpg --env=Half-Cheetah-RM1-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/crm-rs/cheetah/M1/$i --use_crm --use_rs
	python3.6 run.py --alg=dhrm --env=Half-Cheetah-RM1-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/hrm/cheetah/M1/$i --r_max=1000

	# Task 2
	python3.6 run.py --alg=ddpg --env=Half-Cheetah-RM2-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/ql/cheetah/M2/$i
	python3.6 run.py --alg=ddpg --env=Half-Cheetah-RM2-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/ql-rs/cheetah/M2/$i --use_rs
	python3.6 run.py --alg=ddpg --env=Half-Cheetah-RM2-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/crm/cheetah/M2/$i --use_crm
	python3.6 run.py --alg=ddpg --env=Half-Cheetah-RM2-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/crm-rs/cheetah/M2/$i --use_crm --use_rs
	python3.6 run.py --alg=dhrm --env=Half-Cheetah-RM2-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=../my_results/hrm/cheetah/M2/$i --r_max=1000
done
