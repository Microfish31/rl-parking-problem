from highway_env.envs.parking_env import ParkingEnv
import numpy as np
from highway_env.vehicle.objects import Landmark,Obstacle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.graphics import VehicleGraphics
import random

class ParkingWithObstacles(ParkingEnv):
    def __init__(self, env):
        # Set subclass attributes before initializing the parent class
        self.env = env
        self.open_walls = False # set the wall around the parking lot
        self.num_obstacles = 0  # Set number of obstacles
        self.num_init_vehocles = 0 # Set number of initial vehicles
        self.collision_reward = -5 # set collision reward
        self.success_goal_reward = 0.12 # set goal reward
        self.further_reward = -10 # not yet (may be not needed)
        self.duration = 6

        # observation
        ## type
        ## KinematicsGoal indicates that the observation type is based on kinematics,
        ## meaning the observation will include information about the object's motion state
        
        ## feature (6)
        ##  x: The object's x-coordinate (position)
        ##  y: The object's y-coordinate (position)
        ##  vx: The object's velocity along the x-axis
        ##  vy: The object's velocity along the y-axis
        ##  cos_h: The cosine of the object's heading (used to represent its direction)
        ##  sin_h: The sine of the object's heading (also used to represent its direction)
        
        ## scale
        ## 100, meaning the position values are scaled by a factor of 100.
        ## 5, meaning the velocity values are scaled by a factor of 5.
        ## 1, meaning the direction values are not scaled.

        ## normalize
        ## normalize determines whether the observation values should be normalized. 
        ## If set to True, the observations would be adjusted to fit within a specific range, 
        ## such as [0, 1]

        # action (5)
        ## type
        ## DiscreteMetaAction means that actions is not continuous.
        ## 1. LANE_LEFT
        ## 2. IDLE
        ## 3. LANE_RIGHT
        ## 4. FASTER
        ## 5. SLOWER
        ##
        ## examples:
        ## ACTIONS_ALL = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
        
        config = {
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": True
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "screen_height": 600,
            "screen_width": 1200,
            "vehicles_count": self.num_init_vehocles,  # No initial vehicles
            "add_walls": self.open_walls,   # Turn off automatic walls, as we'll add obstacles manually
            "add_obstacles": True,
            "obstacles_count": self.num_obstacles, 
            "centering_position": [0.5, 0.5],
            "scaling": 9,
            "controlled_vehicles": 1,
            "collision_reward": self.collision_reward,
            "success_goal_reward": self.success_goal_reward,
            "reward_weights": [1, 0.3, 0, 0, 0.9, 0.9],
            "duration": self.duration, # The episode is truncated if the time is over. (steps)
        }

        # Initialize the parent class
        super().__init__(config=config, render_mode=env.render_mode)

        # Set observation space and action space to match the original environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        empty_spots = list(self.road.network.lanes_dict().keys())

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            x0 = random.uniform(-20,20)
            y0 = random.uniform(-10,10)
            vehicle = self.action_type.vehicle_class(
                self.road, [x0, y0], 2 * np.pi * self.np_random.uniform(), 0
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            empty_spots.remove(vehicle.lane_index)

        # Goal
        for vehicle in self.controlled_vehicles:
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        # The other goals

        # Other vehicles
        for i in range(self.config["vehicles_count"]):
            if not empty_spots:
                continue
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            v = Vehicle.make_on_lane(self.road, lane_index, 4, speed=0)
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        # Walls
        if self.config["add_walls"]:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

        # Obstacles
        if self.config["add_obstacles"]:
            for i in range(self.config["obstacles_count"]):
                lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
                lane = self.road.network.get_lane(lane_index)
                position = lane.position(lane.length / 2, 0)
                position[1] = round(random.uniform(0, position[1]))
                size = 2  # Set the side length of square obstacles
                # Create a square obstacle in the center of the map
                obstacle = Obstacle(self.road, position)  # Place it at the center of the map
                obstacle.LENGTH = obstacle.WIDTH = size  # Set to square
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
                empty_spots.remove(lane_index)

    def _reward(self, action: np.ndarray) -> float:
        # Get an observation of the environment state.
        obs = self.observation_type_parking.observe()
        # check obs is tuple or not, if not (obs,)
        obs = obs if isinstance(obs, tuple) else (obs,)

        # diff between observation["achieved_goal"] and observation["desired_goal"] , then compute reward
        ## 2 case
        ## 1. Further from the parking space
        ## 2. Stop at parking space 
        reward = sum(
            self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            for agent_obs in obs
        )

        # compute reward when collision
        ## 1 case
        ## 1. Collide with obstacle 
        reward += self.config["collision_reward"] * sum(
            v.crashed for v in self.controlled_vehicles
        )
        return reward