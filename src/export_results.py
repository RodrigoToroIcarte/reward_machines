import math,os,statistics,argparse
from collections import deque
import numpy as np

def get_precentiles_str(a):
    p25 = "%0.2f"%float(np.percentile(a, 25))
    p50 = "%0.2f"%float(np.percentile(a, 50))
    p75 = "%0.2f"%float(np.percentile(a, 75))
    return [p25, p50, p75]


def export_avg_results(agent,env,maps,seeds):
    """
    NOTE: 
        - Find a way to summarize the results coming from different seeds
        - This is tricky because the timesteps per trace might be different
    """

    if 'office' in env:
        num_episodes_avg = 400
        num_total_steps = 1e5
        max_length = 100 
    else:
        num_episodes_avg = 1000
        num_total_steps = 2e6
        max_length = 200 


    stats = [[] for _ in range(max_length)]
    for env_map in maps:
        for seed in seeds:
            # Reading the results
            f_path = "results/%s/%s/%s/%s/0.0.monitor.csv"%(agent,env,env_map,seed)
            results = []
            f = open(f_path)
            for l in f:
                raw = l.strip().split(',')
                if len(raw) != 3 or raw[0]=='r':
                    continue
                r,l,t = float(raw[0]), float(raw[1]), float(raw[2])
                results.append((t,l,r))
            f.close()

            # collecting average stats
            steps = 0
            rewards = deque([], maxlen=num_episodes_avg)
            steps_tic = num_total_steps/max_length
            for i in range(len(results)):
                _,l,r = results[i]
                rew_per_step = 10000*r/l
                if (steps+l)%steps_tic == 0:
                    steps += l
                    rewards.append(rew_per_step)
                    stats[int((steps+l)//steps_tic)-1].append(sum(rewards)/len(rewards))
                else:
                    if (steps//steps_tic) != (steps+l)//steps_tic:
                        stats[int((steps+l)//steps_tic)-1].append(sum(rewards)/len(rewards))
                    steps += l
                    rewards.append(rew_per_step)
                if (steps+l)//steps_tic == max_length:
                    break

    # Saving the average performance and standard deviation
    f_out = "results/summary/%s-%s.txt"%(env,agent)
    f = open(f_out, 'w')
    for i in range(max_length):
        if len(stats[i]) == len(seeds) * len(maps):
            f.write("\t".join([str((i+1)*steps_tic/1000), "%0.4f"%(sum(stats[i])/len(stats[i]))]) + "\n")
            #f.write("\t".join([str((i+1)*steps_tic/1000)] + get_precentiles_str(stats[i])) + "\n")
    f.close()

def main():
    export_avg_results('qrm','water_big',['M0'],[0])
    #export_avg_results('ql','water',['M0'],[0])

    #for alg in ['ql', 'qrm', 'hrl', 'rs','qrm-rs']:
    #    print(alg,'craft')
    #    export_avg_results(alg,'craft',['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10'],[0,1,2])

    #for alg in ['ql', 'qrm', 'hrl', 'rs','qrm-rs']:
    #    print(alg,'office')
    #    export_avg_results(alg,'office',['M1'],list(range(30)))


if __name__ == '__main__':
    main()


