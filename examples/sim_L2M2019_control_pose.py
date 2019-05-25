from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np

mode = '2D'
difficulty = 0
visualize=True
seed=None
sim_dt = 0.01
sim_t = 10
timstep_limit = int(round(sim_t/sim_dt))

#params = np.ones(45)
#params = np.loadtxt('./optim_data/cma/trial_190510_L2M2019CtrlEnv_d0_best_w.txt')
#params = np.loadtxt('./optim_data/params_3D_init.txt')
xrecentbest = open("./optim_data/cma/trial_190521_L2M2019CtrlEnv_d0_xrecentbest.dat", "r")
for line in xrecentbest:
    pass
last = np.fromstring(line, sep=' ')
params_all = np.array(last[5:])

l_femur = .43 # !!! find value from model file
l_tibia = .43
h_foot = .1
# init_pose params
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

#np.savetxt('init_pose.txt', init_pose)

locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
env = L2M2019Env(visualize=visualize, seed=seed, difficulty=difficulty)
env.change_model(model=mode, difficulty=difficulty, seed=seed)
obs_dict = env.reset(project=True, seed=seed, obs_as_dict=True)
env.spec.timestep_limit = timstep_limit

total_reward = 0
t = 0
i = 0
params = params_all[8:]
np.savetxt('params_2D.txt', params)
while True:
    i += 1
    t += sim_dt

    # chage params
    # params = myHigherLayeyController(obs_dict)

    locoCtrl.set_control_params(params)
    action = locoCtrl.update(obs_dict)
    obs_dict, reward, done, info = env.step(action, project = True, obs_as_dict=True)
    total_reward += reward
    #import pdb; pdb.set_trace()
    if i%5 is 1:
        print(obs_dict['pelvis']['vel'][0])
    if done:
        break
print('    score={} time={}sec'.format(total_reward, t))