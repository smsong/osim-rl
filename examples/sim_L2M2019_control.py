from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np

mode = '3D'
difficulty = 0
visualize=True
seed=None
sim_dt = 0.01
sim_t = 10
timstep_limit = int(round(sim_t/sim_dt))

if mode is '2D':
    params = np.loadtxt('./osim/control/params_2D.txt')
elif mode is '3D':
    params = np.loadtxt('./osim/control/params_3D_init.txt')

locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
env = L2M2019Env(visualize=visualize, seed=seed, difficulty=difficulty)
env.change_model(model=mode, difficulty=difficulty, seed=seed)
obs_dict = env.reset(project=True, seed=seed, obs_as_dict=True)
env.spec.timestep_limit = timstep_limit

total_reward = 0
t = 0
i = 0
while True:
    i += 1
    t += sim_dt

    locoCtrl.set_control_params(params)
    action = locoCtrl.update(obs_dict)
    obs_dict, reward, done, info = env.step(action, project = True, obs_as_dict=True)
    total_reward += reward
    if done:
        break
print('    score={} time={}sec'.format(total_reward, t))