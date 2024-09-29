# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'l_finger_joint': 0.0,    # [m]
            'r_finger_joint': 0.0,    # [m]
        }

    class init_state_slope( LeggedRobotCfg.init_state ):
        pos = [0.56, 0.0, 0.24] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.03,   # [rad]
            'RL_hip_joint': 0.03,   # [rad]
            'FR_hip_joint': -0.03,  # [rad]
            'RR_hip_joint': -0.03,   # [rad]

            'FL_thigh_joint': 1.0,     # [rad]
            'RL_thigh_joint': 1.9,   # [rad]1.8
            'FR_thigh_joint': 1.0,     # [rad]
            'RR_thigh_joint': 1.9,   # [rad]

            'FL_calf_joint': -2.2,   # [rad]
            'RL_calf_joint': -0.9,    # [rad]
            'FR_calf_joint': -2.2,  # [rad]
            'RR_calf_joint': -0.9,    # [rad]

            'l_finger_joint': 0.0,    # [m]
            'r_finger_joint': 0.0,    # [m]
        }
    class env(LeggedRobotCfg.env):
        num_envs = 8000 #6144
        num_actions = 13
        num_dummy_dof = 1
        
        n_scan = 132
        n_priv = 3+3 +3
        n_priv_latent = 4 + 1 + 14 +14
        n_proprio = 1 + 3 + 2 + 2 + 4 + 13*3 + 4
        history_len = 10

        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        cur_goal_max_time = 5

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192 #192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.3, 0, 0.147]  # front camera
        position_rand = 0.01  
        angle = [29-5, 29+5]  # positive pitch down  #27-5,27+5
        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 86

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values 1.0
        quantize_height = True
        class noise_scales:
            rotation = 0.1
            dof_pos = 0.005
            dof_vel = 0.075
            # lin_vel = 0.05
            ang_vel = 0.15
            # gravity = 0.02
            height_measurements = 0.02
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2g_description_v6.urdf'
        foot_name = "foot"
        finger_name = "finger"
        penalize_contacts_on = ["finger", "gripper"]  #["base", "thigh", "calf", "finger", "gripper"]
        terminate_after_contacts_on = ["base","finger","gripper"]#,"thigh","finger", "gripper"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand):
        friction_range = [0.2, 2.]  # 0.6 2 6?  0.6-8
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        max_push_vel_z = 0.5 #0.1

        delay_update_global_steps = 24 * 2000  #8000 3000
        action_delay = True
        action_curr_step = [1,1] #[0, 1, 2, 0, 1]
        action_curr_step_scratch = action_curr_step #[0, 1]
        action_delay_view = 1
        action_buf_len = 8
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            tracking_goal_vel = 1.5
            
            tracking_lin_vel = 0.5
            tracking_yaw_vel = 1.
            tracking_pitch = 1.5
            tracking_gripper = 0.5

            # regularization rewards
            lin_vel_z_walking = -9.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            collision = -5.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            hip_pos = -1
            dof_error = -0.2
            feet_stumble = -1
            feet_edge = -1
            feet_drag = -0.1
            energy = -1e-3
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        print_rewards = False

    class terrain( LeggedRobotCfg.terrain):
        terrain_dict = {
                "parkour_flat": 0.5,
                "parkour_step": 0.5,}
        terrain_proportions = list(terrain_dict.values())
        y_range = [-0.1, 0.1]
        cur_threshold_hi = 0.8
        cur_threshold_lo = 0.5  # 3
        
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall ! 10
        num_cols = 40 # number of terrain cols (types)  40
        measured_edge_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_edge_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        
        curriculum = True

    class commands( LeggedRobotCfg.commands):
        num_commands = 4 # default: lin_vel_x, lin_vel_y, omega, pitch
        class max_ranges:
            lin_vel_x = [-0.5*0, 1.] # min max [m/s] -0.5
            lin_vel_x_parkour = [0.5, 1.2] # min max [m/s]  # 0.5 1.2
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]0.5
            omega = [-0.7, 0.7]    # min max [rad/s]
            pitch = [-0.02, 0.02]  # min max [rad]
        lin_vel_clip = 0.02  #0.2 0.02 0.05
        ang_clip = 0.05
        pitch_clip = 0.05 

    class sim (LeggedRobotCfg.sim):
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.01   # 0.0 [m] 0.01
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
        
class Go2RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01  # 0 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2'
        resume = False
        max_iterations = 50000 # number of policy updates 50000

    class estimator( LeggedRobotCfgPPO.estimator):
        priv_states_dim = Go2RoughCfg.env.n_priv
        num_prop = Go2RoughCfg.env.n_proprio
  