import gym
import numpy as np
import PIL.Image
import PIL.ImageDraw
import copy
import matplotlib.pyplot as plt
from config import config

car_img = PIL.Image.open('assets/car.png')
car_img = car_img.convert('RGBA')
car_img = car_img.resize((car_img.size[0]//5, car_img.size[1]//5))

goal_img = PIL.Image.open('assets/goal.png')
goal_img = goal_img.convert('RGBA')
goal_img = goal_img.resize((goal_img.size[0]//10, goal_img.size[1]//10))
# plt.imshow(car_img)
# plt.show()

class GoalBasedKBM:

    # 2D planar navigation task with changing goals
    # Rules:
    # 1) Reacing the goal results in a positive reward
    # 2) You die if you go outside the box, or after timeout
    # 3) Constant negative reward for every other timestep in the environment
    MIN_POS = 0.  # if box exists
    MAX_POS = 10.  # in meters
    # g*sinθ -> Implicitly also captures the slope of the plane
    SLOPE_X = 0.
    SLOPE_Y = 0.
    GRAVITATIONAL_CONSTANT = 9.81
    simulator_frequency = 10
    controller_frequency = 10

    def __init__(self):
      self.dt = 1 / self.simulator_frequency
      self.length = 0.25 # vehicle's longitudinal length
      self.goal_thresh = 0.5
      self.angle_thresh = 10 * np.pi / 180
      self.eval_flag = False
      self.fixed_goal = False
      self.sparse_reward_flag = False
      self.yaw_alignment_penalty = True

      # state: [x, y, yaw, speed, steering]
      # action: [acceleration, steering_rate]
      self.action_repeat_count = self.simulator_frequency // self.controller_frequency
      self.min_action = np.array([-1., -1.], dtype=np.float32)
      self.max_action = np.array([1., 1.], dtype=np.float32)
      self.min_obs = np.array([self.MIN_POS, self.MIN_POS, -np.pi, 0, -0.611, -np.pi, -1, -1], dtype=np.float32)
      self.max_obs = np.array([self.MAX_POS, self.MAX_POS, np.pi, 10, 0.611, np.pi, 1, 1], dtype=np.float32)
      self.action_space = gym.spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)
      self.observation_space = gym.spaces.Box(low=self.min_obs, high=self.max_obs, dtype=np.float32)
      
      self.state = None
      self.goal = None
      self.episode_steps = None
      self.adaptive_max_steps = None
      self.dist_2_goal = None
      self.last_yaw_error = None
      self.sum_of_dists = None
      self.metadata = {"render_modes": ["human", "rgb_array based off of master, and put your code in the correct spot."], "render_fps": 30}
      # self.reset()

    def effective_gravity(self):
      dGx = self.SLOPE_X
      dGy = self.SLOPE_Y
      yaw = self.state[2]
      
      x_comp = ((dGy**2 + 1) * np.cos(yaw) - dGx * dGy * np.sin(yaw)) # / (dGx**2 + dGy**2 + 1)
      y_comp = ((dGx**2 + 1) * np.sin(yaw) - dGx * dGy * np.cos(yaw)) # / (dGx**2 + dGy**2 + 1)
      z_comp = (dGx * np.cos(yaw) + dGy * np.sin(yaw)) # / (dGx**2 + dGy**2 + 1)
      
      normed_z_comp = z_comp / np.sqrt(x_comp**2 + y_comp**2 + z_comp**2 + 1e-8)

      return -self.GRAVITATIONAL_CONSTANT * normed_z_comp

    def reset(self):
      self.state = np.zeros(5)
      self.goal = np.zeros(3)
      
      if self.fixed_goal:
        self.goal = np.array([0.9*self.MAX_POS, 0.9*self.MAX_POS, np.pi/4])
        self.state[:2] = np.array([0.5*self.MAX_POS,0.5*self.MAX_POS])
        self.state[2] = np.pi/4
      else:
        self.state[:2] = np.random.uniform(self.MIN_POS + 0.5, self.MAX_POS - 0.5, size=(2)) # XY
        self.goal[:2] = np.random.uniform(self.MIN_POS + 0.5, self.MAX_POS - 0.5, size=(2)) # XY
        
        self.state[2] = np.arctan2(self.goal[1] - self.state[1], self.goal[0] - self.state[0])
        self.goal[2] = self.state[2] + np.random.uniform(-np.pi/4, np.pi/4)
        self.state[2] += np.random.uniform(-np.pi/4, np.pi/4)
        # self.state[2] = np.random.uniform(-np.pi, np.pi) # Yaw
        # self.goal[2] = np.random.uniform(-np.pi, np.pi) # Goal Yaw
      
      # A heuristic for early termination:
      # Time taken to cover target_displacement in 1m/s
      # Note: Distance covered is usually > 2x displacement since robot is non-holonomic
      self.adaptive_max_steps = 5 * self.controller_frequency * np.linalg.norm(self.goal[:2] - self.state[:2])
      self.adaptive_max_steps = max(self.adaptive_max_steps, 40)
      self.episode_steps = 0
      self.dist_2_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
      self.sum_of_dists = np.linalg.norm(self.state[:2] - self.goal[:2])
      if self.yaw_alignment_penalty:
        del_x = self.goal[0] - self.state[0]
        del_y = self.goal[1] - self.state[1]
        del_x, del_y = del_x * np.cos(self.state[2]) + del_y * np.sin(self.state[2]), -del_x * np.sin(self.state[2]) + del_y * np.cos(self.state[2])
        del_yaw = self.goal[2] - self.state[2]
        
        # Find the circle that passes tangentially through the goal and (0,0)
        # We only need denominator for its sign, since arctan2 is sensitive to sign of (dy, dx)
        denominator = 2*(del_y - del_x*np.tan(del_yaw))
        center_x = ((del_y**2 - del_x**2)*np.tan(del_yaw) + 2*del_x*del_y) / denominator
        center_y = ((del_y**2 - del_x**2) - 2*del_x*del_y*np.tan(del_yaw)) / denominator
        
        direction = (del_y - center_y)*np.cos(del_yaw) - (del_x - center_x)*np.sin(del_yaw)
        desired_yaw = np.arctan2(-center_x, center_y)
        if direction > 0:
          desired_yaw = self.warp2pi(desired_yaw - np.pi)
        # current yaw is 0 in vehicle-centric frame
        self.last_yaw_error = config['yaw_penalty'] * np.abs(desired_yaw)
      
      delta_angle = np.arctan2(self.goal[1]-self.state[1], self.goal[0]-self.state[0]) - self.state[2]
      delta_angle = self.warp2pi(delta_angle)
      rot_matrix = np.array([[np.cos(self.state[2]), np.sin(self.state[2])],
                            [-np.sin(self.state[2]), np.cos(self.state[2])]])
      delta_xy = rot_matrix @ (self.goal[:2] - self.state[:2])
      assert (np.arctan2(delta_xy[1], delta_xy[0]) - delta_angle)**2 < 1e-4
      
      target_yaw = self.warp2pi(self.goal[2] - self.state[2])
      target_yaw_repr = [target_yaw, np.sin(target_yaw), np.cos(target_yaw)]
      return np.concatenate([delta_xy, [delta_angle], self.state[3:], target_yaw_repr])

    def warp2pi(self, angle_rad):
      """
      warps an angle in [-pi, pi]
      """
      res = (angle_rad)%(2*np.pi)
      if res <= np.pi:
          return res
      return res - 2*np.pi

    def step(self, action):
      assert len(action) == 2 # Be cautious when using DummyVecEnv <-> actions also get vectorized
      action = np.minimum(np.maximum(action, self.min_action), self.max_action)
    
      for _ in range(max(self.action_repeat_count, 1)):
        action[0] += self.effective_gravity()
        d_system_state = np.zeros(5)
        d_system_state[0] = (self.state[3]*self.dt + 0.5*action[0]*self.dt**2)*np.cos(self.state[2])
        d_system_state[1] = (self.state[3]*self.dt + 0.5*action[0]*self.dt**2)*np.sin(self.state[2])
        d_system_state[2] = (self.state[3]*self.dt + 0.5*action[0]*self.dt**2)*np.tan(self.state[4])/self.length
        d_system_state[3] = action[0]*self.dt
        d_system_state[4] = action[1]*self.dt

        self.state = self.state + d_system_state
        self.state[2] = self.warp2pi(self.state[2])
        self.state[3:] = np.minimum(np.maximum(self.state[3:5], self.min_obs[3:5]), self.max_obs[3:5])

      if self.yaw_alignment_penalty:
        del_x = self.goal[0] - self.state[0]
        del_y = self.goal[1] - self.state[1]
        del_x, del_y = del_x * np.cos(self.state[2]) + del_y * np.sin(self.state[2]), -del_x * np.sin(self.state[2]) + del_y * np.cos(self.state[2])
        del_yaw = self.warp2pi(self.goal[2] - self.state[2])
        
        # Find the circle that passes tangentially through the goal and (0,0)
        # We only need denominator for its sign, since arctan2 is sensitive to sign of (dy, dx)
        denominator = 2*(del_y - del_x*np.tan(del_yaw))
        center_x = ((del_y**2 - del_x**2)*np.tan(del_yaw) + 2*del_x*del_y) / denominator
        center_y = ((del_y**2 - del_x**2) - 2*del_x*del_y*np.tan(del_yaw)) / denominator
        
        direction = (del_y - center_y)*np.cos(del_yaw) - (del_x - center_x)*np.sin(del_yaw)
        desired_yaw = np.arctan2(-center_x, center_y)
        if direction > 0:
          desired_yaw = self.warp2pi(desired_yaw - np.pi)
        
        direction2 = (-center_y)*np.cos(desired_yaw) - (-center_x)*np.sin(desired_yaw)
        assert (direction > 0) == (direction2 > 0)
      
      # if the sim goes really fast, they can bounce one-step out of box. Let's just check for this for now, fix later
      info = {}
      # if not np.all(self.state[:2] >= self.MIN_POS) and np.all(self.state[:2] <= self.MAX_POS):
      #   print("Out-of-Bounds Termination. Resetting....")
      #   reward, done = -10, True
      if np.linalg.norm(self.state[:2] - self.goal[:2]) < self.goal_thresh:
        if np.abs(self.state[2] - self.goal[2]) < self.angle_thresh:
          print("Goal Reached with CORRECT YAW. Resetting.... (Angle Error: {})".format(
                  180 * np.abs(self.warp2pi(self.state[2] - self.goal[2])) / np.pi))
          reward, done = 100, True
        else:
          print("Goal Reached with INCORRECT YAW. Resetting.... (Angle Error: {})".format(
                  180 * np.abs(self.warp2pi(self.state[2] - self.goal[2])) / np.pi))
          reward, done = 10, True
      elif self.episode_steps > self.adaptive_max_steps:
        print("Timeout Termination. Resetting....")
        info["TimeLimit.truncated"] = True
        # info["terminal_observation"] = copy.deepcopy(self.state) # Automatically set
        reward, done = -5, True
      elif self.sparse_reward_flag:
        reward, done = -1, False
      else:
        reward = -1 + self.dist_2_goal - config['gamma'] * np.linalg.norm(self.state[:2] - self.goal[:2])
        if self.yaw_alignment_penalty:
          # current yaw is 0 in vehicle-centric frame
          reward += self.last_yaw_error - config['yaw_penalty'] * config['gamma'] * np.abs(desired_yaw)
        done = False
      
      self.dist_2_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
      if self.yaw_alignment_penalty:
        self.last_yaw_error = config['yaw_penalty'] * np.abs(desired_yaw)
      self.sum_of_dists += self.dist_2_goal

      # self.state = np.minimum(np.maximum(self.state, self.min_obs[:5]), self.max_obs[:5])
      self.episode_steps += 1

      if self.eval_flag:
        obs_img = self.render()
        plt.ion()
        plt.clf()
        plt.imshow(obs_img)
        plt.show()
        plt.pause(0.01)
      
      delta_angle = np.arctan2(self.goal[1]-self.state[1], self.goal[0]-self.state[0]) - self.state[2]
      delta_angle = self.warp2pi(delta_angle)
      rot_matrix = np.array([[np.cos(self.state[2]), np.sin(self.state[2])],
                            [-np.sin(self.state[2]), np.cos(self.state[2])]])
      delta_xy = rot_matrix @ (self.goal[:2] - self.state[:2])
      assert (np.arctan2(delta_xy[1], delta_xy[0]) - delta_angle)**2 < 1e-4
      
      target_yaw = self.warp2pi(self.goal[2] - self.state[2])
      target_yaw_repr = [target_yaw, np.sin(target_yaw), np.cos(target_yaw)]
      return np.concatenate([delta_xy, [delta_angle], self.state[3:], target_yaw_repr]), reward, done, info

    def render(self):
      size = 1000
      radius = 0.1
      img = PIL.Image.new('RGBA', (size, size), (255,255,255))

      def coord_transform(xy):
        new_coord = copy.deepcopy(xy[::-1])
        new_coord[0] = self.MAX_POS - new_coord[0]
        new_coord = new_coord[::-1]
        return new_coord
      
      coords = size * np.tile(coord_transform(self.goal[:2]), 2) / self.MAX_POS
      draw = PIL.ImageDraw.Draw(img, 'RGBA')
      draw.ellipse(tuple(coords + np.array([-self.goal_thresh, -self.goal_thresh, self.goal_thresh, self.goal_thresh])*size/self.MAX_POS), fill=(0,255,0, 25), width=10, outline ="green")
      
      rot_goal_img = goal_img.rotate(self.goal[2]*180/np.pi, PIL.Image.BICUBIC, fillcolor=(255, 255, 255), expand = 1)
      goal_mask = PIL.Image.new("L", rot_goal_img.size, 0)
      draw_mask = PIL.ImageDraw.Draw(goal_mask)
      trim = 0
      bounds = ((trim,trim), tuple((np.array(rot_goal_img.size)-trim).astype(int)))
      draw_mask.rectangle(bounds, fill=255)
      goal_loc = coord_transform(self.goal[:2])
      img.paste(rot_goal_img, tuple((size*goal_loc/self.MAX_POS - np.array(rot_goal_img.size)/2).astype(int)), goal_mask)
      
      rot_car_img = car_img.rotate(self.state[2]*180/np.pi, PIL.Image.BICUBIC, fillcolor=(255, 255, 255), expand = 1)
      mask = PIL.Image.new("L", rot_car_img.size, 0)
      draw_mask = PIL.ImageDraw.Draw(mask)
      trim = 0
      bounds = ((trim,trim), tuple((np.array(rot_car_img.size)-trim).astype(int)))
      draw_mask.rectangle(bounds, fill=255)
      # from PIL import ImageFilter
      # mask_im_blur = mask.filter(ImageFilter.GaussianBlur(10))
      
      center = coord_transform(self.state[:2]) # np.array([0.5,0.5])
      img.paste(rot_car_img, tuple((size*center/self.MAX_POS - np.array(rot_car_img.size)/2).astype(int)), mask)
      
      import cv2
      na = np.array(img)
      # Draw arrowed line, from 10,20 to w-40,h-60 in black with thickness 8 pixels
      na = cv2.arrowedLine(na, (coords[:2]-self.goal_thresh).astype(int), (coords[:2]+self.goal_thresh).astype(int), (255,255,255), 8)
      # Revert back to PIL Image and save
      img = PIL.Image.fromarray(na)
      
      img = img.resize((300, 300))
      img = np.array(img)
      return img
