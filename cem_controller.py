import numpy as np
from config import config
import matplotlib.pyplot as plt
from dynamics_model import GoalBasedKBM

eval_flag = False
fixed_goal = False
sparse_reward_flag = False

env = GoalBasedKBM()
env.eval_flag = eval_flag
env.fixed_goal = fixed_goal
env.sparse_reward_flag = sparse_reward_flag

dt = 0.1
gamma = config['gamma']
goal_thresh = 0.5
angle_thresh = 10 * np.pi / 180
horizon = 50
cem_iters = 10
elite_fraction = 1/8
sample_size = 1024
start_std = 1.
vehicle_length = 0.25
mppi_temperature = 10
min_std = 0.1

np.random.seed(10)

elite_nums = int(sample_size * elite_fraction)

def warp2pi(angle_rad):
      """
      warps an angle in [-pi, pi]
      """
      res = (angle_rad)%(2*np.pi)
      res = (res <= np.pi)*res + (res > np.pi)*(res - 2*np.pi)
      return res

def angle_penalty(state, goal_yaw):
    # Find the circle that passes tangentially through the goal and (0,0)
    del_x = state[:, 5] - state[:, 0]
    del_y = state[:, 6] - state[:, 1]
    del_x, del_y = del_x * np.cos(state[:,2]) + del_y * np.sin(state[:,2]), -del_x * np.sin(state[:,2]) + del_y * np.cos(state[:,2])
    del_yaw = warp2pi(goal_yaw - state[:, 2])
    
    # We only need denominator for its sign, since arctan2 is sensitive to sign of (dy, dx)
    denominator = 2*(del_y - del_x*np.tan(del_yaw))
    center_x = ((del_y**2 - del_x**2)*np.tan(del_yaw) + 2*del_x*del_y) / denominator
    center_y = ((del_y**2 - del_x**2) - 2*del_x*del_y*np.tan(del_yaw)) / denominator
    
    direction = (del_y - center_y)*np.cos(del_yaw) - (del_x - center_x)*np.sin(del_yaw)
    desired_yaw = np.arctan2(-center_x, center_y)
    desired_yaw = (direction <= 0) * desired_yaw + (direction > 0) * warp2pi(desired_yaw - np.pi)
    
    # current yaw is 0 in vehicle-centric frame
    return np.abs(desired_yaw)

def kbm_rollouts(states, actions, goal_yaw):
    not_done = np.ones(sample_size)
    rewards_vector, traj_len = np.zeros(sample_size), np.zeros(sample_size)
    for i in range(horizon):
        d_system_state = np.zeros((sample_size, 5))
        d_system_state[:,0] = (states[:,i,3]*dt + 0.5*actions[:,i,0]*dt**2)*np.cos(states[:,i,2])
        d_system_state[:,1] = (states[:,i,3]*dt + 0.5*actions[:,i,0]*dt**2)*np.sin(states[:,i,2])
        d_system_state[:,2] = (states[:,i,3]*dt + 0.5*actions[:,i,0]*dt**2)*np.tan(states[:,i,4])/vehicle_length
        d_system_state[:,3] = actions[:,i,0]*dt
        d_system_state[:,4] = actions[:,i,1]*dt

        states[:,i+1,:5] = states[:,i,:5] + d_system_state
        states[:,i+1,3:5] = np.minimum(np.maximum(states[:,i+1,3:5], np.array([0,-0.611])), np.array([10,0.611]))
        
        d2g_i = np.linalg.norm(states[:,i,:2] - states[:,i,-2:], axis=-1)
        d2g_i1 = np.linalg.norm(states[:,i+1,:2] - states[:,i+1,-2:], axis=-1)
        
        rewards_vector += gamma**i *not_done*(d2g_i - gamma * d2g_i1 - 1)
        
        rewards_vector += (gamma**i *not_done*(angle_penalty(states[:,i], goal_yaw) \
                        - gamma * angle_penalty(states[:,i+1], goal_yaw)))
        
        new_not_done = not_done * (np.linalg.norm(states[:,i+1,:2] - states[:,i+1,-2:], axis=-1) >= goal_thresh)
        
        rewards_vector += (1-new_not_done) * not_done * gamma**(i+1) * (10 + 90*(angle_penalty(states[:,i+1], goal_yaw) < angle_thresh))
        
        not_done = new_not_done
        traj_len += not_done

    return rewards_vector, states, traj_len

def cem_act(state, action_means, goal_yaw):
    action_std = np.ones((horizon, 2))
    for _ in range(cem_iters):
        action_samples = np.random.normal(action_means, action_std, (sample_size, horizon, 2))
        action_samples = np.minimum(np.maximum(action_samples, -1), 1)
        states = np.tile(state[None, None, :], (sample_size, horizon+1, 1))
        rewards_vector, future_states, traj_len = kbm_rollouts(states, action_samples, goal_yaw)
        elite_inds = np.argpartition(rewards_vector, -elite_nums)[-elite_nums:]
        
        # # CEM Algorithm
        # action_means = action_samples[elite_inds].mean(0)
        # action_std = action_samples[elite_inds].std(0)
        
        # MPPI Algorithm
        weights = np.exp(mppi_temperature * (rewards_vector[elite_inds] - rewards_vector[elite_inds].max()))
        weights = weights/ weights.sum()
        action_means = (action_samples[elite_inds] * weights[:, None, None]).sum(0)
        action_std = np.sqrt((weights[:, None, None] * (action_samples[elite_inds] - action_means)**2).sum(0))
        action_std = np.maximum(action_std, min_std)
        
        
    elite_inds = np.argpartition(rewards_vector, -1)[-1]
    traj_len = int(traj_len[elite_inds])
    
    
    del_x = state[5] - state[0]
    del_y = state[6] - state[1]
    del_x, del_y = del_x * np.cos(state[2]) + del_y * np.sin(state[2]), -del_x * np.sin(state[2]) + del_y * np.cos(state[2])
    del_yaw = warp2pi(goal_yaw - state[2])

    denominator = 2*(del_y - del_x*np.tan(del_yaw))
    center_x = ((del_y**2 - del_x**2)*np.tan(del_yaw) + 2*del_x*del_y) / denominator
    center_y = ((del_y**2 - del_x**2) - 2*del_x*del_y*np.tan(del_yaw)) / denominator
    radius = np.sqrt(center_x**2 + center_y**2)
    
    direction = (del_y - center_y)*np.cos(del_yaw) - (del_x - center_x)*np.sin(del_yaw)
    desired_yaw = np.arctan2(-center_x, center_y)
    if direction > 0:
        desired_yaw = warp2pi(desired_yaw - np.pi)
    
    direction2 = (-center_y)*np.cos(desired_yaw) - (-center_x)*np.sin(desired_yaw)
    assert (direction > 0) == (direction2 > 0)
    # desired_yaw = np.abs(desired_yaw)
    # print(180*np.abs(desired_yaw)/np.pi)
    
    angles = np.linspace(-np.pi, np.pi, 100)
    x_vals = center_x + radius * np.cos(angles)
    y_vals = center_y + radius * np.sin(angles)    
    x_vals, y_vals = x_vals * np.cos(state[2]) - y_vals * np.sin(state[2]) + state[0], x_vals * np.sin(state[2]) + y_vals * np.cos(state[2]) + state[1]
    
    plt.xlim(0,300)
    plt.ylim(0,300)
    plt.plot(30*x_vals, 30*(10-y_vals), "--r")
    plt.plot(30*future_states[elite_inds,:traj_len,0], 30*(10-future_states[elite_inds,:traj_len,1]), label="Expected Reward = {}".format(rewards_vector[elite_inds]))
    plt.legend()
    plt.pause(0.1)
    return action_means

plt.ion()
# input("HALTED! GoalBasedKBM Env uses goal yaw but this is not part of the CEM cost function. ENTER to proceed!")
for _ in range(10): # episodes
    _ = env.reset()
    state = np.concatenate([env.state, env.goal[:2]])
    done = False
    action_means = np.zeros((horizon, 2))
    while not done:
        action_means = cem_act(state, action_means, env.goal[2])
        # print(action_means[0])
        _, _, done, _ = env.step(action_means[0])
        state = np.concatenate([env.state, env.goal[:2]])
        action_means = np.roll(action_means, -1)

        img = env.render()
        plt.clf()
        # plt.scatter(state[0], state[1], label="car", c = 'red')
        # plt.scatter(state[-2], state[-1], label="goal", c = 'green')
        # plt.xlim(0,10)
        # plt.ylim(0,10)
        # plt.legend()
        plt.imshow(img)
        # plt.pause(0.5)
        
        if done:
            break