from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
from joblib import Parallel, delayed

import sys
import numpy as np

trial_name = 'trial_190515_L2M2019CtrlEnv_d0_'

#params_all0 = np.ones(46+8)
params_all0 = np.ones(37+8)
#params_all0 = np.loadtxt('./optim_data/cma/trial_181029_walk_3D_noStand_8_best.txt')
N_POP = 18 # 8 = 4 + floor(3*log(37+8)) # 9 = 4 + floor(3*log(45+8))
N_PROC = 2
TIMEOUT = 10*60
      
l_femur = .43 # !!! find value from model file
l_tibia = .43
h_foot = .1

def f_ind(n_gen, i_worker, params_all):
    flag_model = '2D'
    flag_ctrl_mode = '2D'
    seed = None
    difficulty = 0
    sim_dt = 0.01
    sim_t = 10
    timstep_limit = int(round(sim_t/sim_dt))

    init_error = True
    error_count = 0
    while init_error:
        try:
            # init_pose
            params_init = params_all[0:8]
            vx = np.clip(1.5+params_init[0], 1.0, 1.7)
            theta = np.clip(10+params_init[1], 0, 30)*np.pi/180
            r_hip_abd = np.clip(-3+params_init[2], -10, 10)*np.pi/180
            r_alpha = np.clip(70+params_init[3], 50, 90)*np.pi/180
            r_len = np.clip(-.02+.01*+params_init[4], -.1, 0) + l_femur + l_tibia
            l_hip_abd = np.clip(-3+params_init[5], -10, 10)*np.pi/180
            l_alpha = np.clip(120+params_init[6], 100, 150)*np.pi/180
            l_len = np.clip(l_femur + l_tibia - .07 +.01*params_init[0], .6, r_len*np.sin(r_alpha)/np.sin(l_alpha)-.15)

            r_knee = -(np.pi - np.arccos( (l_femur**2 + l_tibia**2 - r_len**2) / (2*l_femur*l_tibia) ))
            l_knee = -(np.pi - np.arccos( (l_femur**2 + l_tibia**2 - l_len**2) / (2*l_femur*l_tibia) ))

            init_pose = np.zeros(11)
            init_pose[0] = vx
            init_pose[1] = r_len*np.sin(r_alpha) + h_foot
            init_pose[2] = theta
            init_pose[3] = r_hip_abd
            init_pose[4] = -theta - (.5*np.pi - r_alpha - .5*r_knee)
            init_pose[5] = r_knee
            init_pose[6] = .5*np.pi - r_alpha + .5*r_knee
            init_pose[7] = l_hip_abd
            init_pose[8] = -theta - (.5*np.pi - l_alpha - .5*l_knee)
            init_pose[9] = l_knee
            init_pose[10] = np.clip(.5*np.pi - l_alpha + .5*l_knee, -10*np.pi/180, 30*np.pi/180)

            locoCtrl = OsimReflexCtrl(mode=flag_ctrl_mode, dt=sim_dt)
            env = L2M2019Env(seed=seed, difficulty=difficulty, visualize=False)
            env.change_model(model=flag_model, difficulty=difficulty, seed=seed)
            obs_dict = env.reset(project=True, seed=seed, init_pose=init_pose, obs_as_dict=True)
            init_error = False
        except Exception as e_msg:
            error_count += 1
            print('\ninitialization error (x{})!!!'.format(error_count))
            #print(e_msg)
            #import pdb; pdb.set_trace()
    env.spec.timestep_limit = timstep_limit

    total_reward = 0
    error_sim = 0;
    t = 0
    params = params_all[8:]
    while True:
        t += sim_dt

        locoCtrl.set_control_params(params)
        action = locoCtrl.update(obs_dict)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=True)
        total_reward += reward

        if done:
            break

    print('\n    gen#={} sim#={}: score={} time={}sec #step={}'.format(n_gen, i_worker, total_reward, t, env.footstep['n']))

    return total_reward  # minimization


class CMATrainPar(object):
    def __init__(self, ):
        self.n_gen = 0
        self.best_total_reward = -np.inf

    def f(self, v_params_all):
        self.n_gen += 1
        timeout_error = True
        error_count = 0
        while timeout_error:
            try:
                v_total_reward = Parallel(n_jobs=N_PROC, timeout=TIMEOUT)\
                (delayed(f_ind)(self.n_gen, i, p) for i, p in enumerate(v_params_all))
                timeout_error = False
            except Exception as e_msg:
                error_count += 1
                print('\ntimeout error (x{})!!!'.format(error_count))
                #print(e_msg)

        for total_reward, i_params_all in zip(v_total_reward, v_params_all):
            if self.best_total_reward  < total_reward:
                filename = "./optim_data/cma/" + trial_name + "best_w.txt"
                print("\n")
                print("----")
                print("update the best score!!!!")
                print("\tprev = %.8f" % self.best_total_reward )
                print("\tcurr = %.8f" % total_reward)
                print("\tsave to [%s]" % filename)
                print("----")
                print("")
                self.best_total_reward  = total_reward
                np.savetxt(filename, i_params_all)

        return [-r for r in v_total_reward]

if __name__ == '__main__':
    prob = CMATrainPar()

    from cmaes.solver_cma import CMASolverPar
    solver = CMASolverPar(prob)

    solver.options.set("popsize", N_POP)
    solver.options.set("maxiter", 400)
    solver.options.set("verb_filenameprefix", 'optim_data/cma/' + trial_name)
    solver.set_verbose(True)

    x0 = params_all0
    sigma = .1

    res = solver.solve(x0, sigma)
    print(res)
