# Import important libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from kinematics import *
from sensors import *
from utils import *
from LQG_QKF.LQG_QKF import LQG
from time import sleep
import tqdm
from typing import Literal
show_animation = True
 
def closed_loop_prediction(desired_traj, landmarks, filter:Literal['EKF', 'KF']='EKF', noise_coeff=1, dt=0.1):
 
    ## Simulation Parameters
    T = desired_traj.shape[0]  # Maximum simulation time
    goal_dis = 0.1 # How close we need to get to the goal
    goal = desired_traj[-1,:] # Coordinates of the goal
    time_0 = 0.0 # Starting time
 
 
    ## Initial States 
    state = np.array([8.3,0.69]) # Initial state of the car
    state_est = state.copy()
 
    ## Get the Cost-to-go and input cost matrices for LQR
    Q = get_Q() # Defined in kinematics.py
    R = get_R() # Defined in kinematics.py
 
    ## Initialize the Car and the Car's landmark sensor 
    DiffDrive = DifferentialDrive(noise_coeff=noise_coeff)
    LandSens = LandmarkDetector(landmarks, noise_coeff=noise_coeff)
         
    # Process noise
    V = DiffDrive.get_V()
    # Sensor measurement noise
    W = LandSens.get_W()
    
    ## Create objects for storing states and estimated state
    t = [time_0]
    traj = np.array([state])
    traj_est = np.array([state_est])
 
    ind = 0
    
    # state dynamics
    A, B = DiffDrive.linearize(dt)
    
    # set up the LQG controller
    lqg = LQG(F=DiffDrive, x_0=state, x_hat_0=state_est, A=A, B=B, 
              sensor=LandSens, Q=Q, R=R, W=W, V=V, H=25, dt=dt, filter=filter)
    
    measurement_err = []
    cost_to_go = []
    
    time_steps = np.arange(time_0, T, dt)
    for time in tqdm.tqdm(time_steps):
         
        ## Point to track
        ind = int(np.floor(time))
        goal_i = desired_traj[ind,:]
 
        ## Generate noise
        # v = process noise, w = measurement noise
        lqg.update_noise()
        
        ## Generate optimal control commands
        lqg.update_lqr(goal_state=goal_i)
 
        ## Move forwad in time
        lqg.forward_state()
        
        # Update the estimate of the state using the EKF
        lqg.update_lqe()
 
        # Store the trajectory and estimated trajectory
        t.append(time)
        state = lqg.x
        state_est = lqg.x_hat
        measure_covar_matrix = lqg.P_lqe
        cost_to_go_matrix = lqg.P_lqr
        # print('error:', (measure_covar_matrix))  
        
        traj = np.concatenate((traj,[state]),axis=0)
        traj_est = np.concatenate((traj_est,[state_est]),axis=0)
        measurement_err.append(np.trace(measure_covar_matrix))
        cost_to_go.append((state_est.T @ cost_to_go_matrix @ state_est).item())
 
        # Check to see if the robot reached goal
        # if np.linalg.norm(state[0:2]-goal[0:2]) <= goal_dis:
        #     print("Goal reached")
        #     break
 
        ## Plot the vehicles trajectory
        if time % 1 < 0.1 and show_animation:
            plt.cla()
            plt.plot(desired_traj[:,0], desired_traj[:,1], "-r", label="course")
            plt.plot(traj[:,0], traj[:,1], "ob", label="trajectory")
            plt.plot(traj_est[:,0], traj_est[:,1], "sk", label="estimated trajectory")

            plt.plot(goal_i[0], goal_i[1], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("SINGAPORE GRAND PRIX\n" + "speed[m/s]:" + str(
                                            round(np.mean(lqg.u), 2)) +
                      ",target index:" + str(ind))
            plt.pause(0.0001)
 
        #input()
    return traj, traj_est, measurement_err, cost_to_go, time_steps
 
 
def main():
    # Create the track waypoints
    ax = [8.3,8.0, 7.2, 6.5, 6.2, 6.5, 1.5,-2.0,-3.5,-5.0,-7.9,
       -6.7,-6.7,-5.2,-3.2,-1.2, 0.0, 0.2, 2.5, 2.8, 4.4, 4.5, 7.8, 8.5, 8.3]
    ay = [0.7,4.3, 4.5, 5.2, 4.0, 0.7, 1.3, 3.3, 1.5, 3.0,-1.0,
       -2.0,-3.0,-4.5, 1.1,-0.7,-1.0,-2.0,-2.2,-1.2,-1.5,-2.4,-2.7,-1.7,-0.1]
     
    # These landmarks help the mobile robot localize itself
    landmarks = [[4,3],
                 [8,2],
                 [-1,-4]]
         
    # Compute the desired trajectory
    desired_traj = compute_traj(ax,ay)
    noise = 0.1
    dt = 0.1
    pred_traj_ukf, est_traj_ekf, err_ukf, cost_ukf, time_steps = closed_loop_prediction(desired_traj,landmarks, filter='UKF', noise_coeff=noise, dt=dt)
    pred_traj_ekf, est_traj_ukf, err_ekf, cost_ekf, time_steps = closed_loop_prediction(desired_traj,landmarks, filter='EKF', noise_coeff=noise, dt=dt)
    # Compute the distance between the desired trajectory and the estimated trajectory
    pred_traj_ukf = pred_traj_ukf[1:]
    pred_traj_ekf = pred_traj_ekf[1:]
    dist_ekf = [math.sqrt((pred_traj_ekf[i,0] - est_traj_ekf[i,0])**2 + (pred_traj_ekf[i,1] - est_traj_ekf[i,1])**2) for i in range(len(pred_traj_ekf))]
    dist_ukf = [math.sqrt((pred_traj_ukf[i,0] - est_traj_ukf[i,0])**2 + (pred_traj_ukf[i,1] - est_traj_ukf[i,1])**2) for i in range(len(pred_traj_ukf))]
    
    # Display the trajectory that the mobile robot executed
    plot_dir = './plots/'
    import os
    os.makedirs(plot_dir, exist_ok=True)
    
    if show_animation:
        plt.close()
        flg, axes = plt.subplots(4,1, figsize=(10, 15))
        # tracking paths
        axes[0].plot(ax, ay, "xb", label="input")
        # half transparency
        axes[0].plot(desired_traj[:,0], desired_traj[:,1], label="spline", color = 'green')
        axes[0].plot(est_traj_ekf[:,0], est_traj_ekf[:,1], label="tracking_ekf", color='blue', alpha=0.5)
        axes[0].plot(est_traj_ukf[:,0], est_traj_ukf[:,1], label="tracking_ukf", color='orange', alpha=0.5)
        axes[0].grid(True)
        axes[0].axis("equal")
        axes[0].set_xlabel("x[m]")
        axes[0].set_ylabel("y[m]")
        axes[0].legend()
        # error
        axes[1].plot(time_steps[:len(err_ekf)], err_ekf, label="error_ekf", color='blue')
        axes[1].plot(time_steps[:len(err_ukf)], err_ukf, label="error_ukf", color='orange')
        axes[1].grid(True)
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("measurement error")
        axes[1].legend()
        # residual distance
        axes[2].plot(time_steps[:len(est_traj_ekf)], dist_ekf, label="distance_ekf", color='blue')
        axes[2].plot(time_steps[:len(est_traj_ukf)], dist_ukf, label="distance_ukf", color='orange')
        axes[2].set_xlabel("time")
        axes[2].set_ylabel("distance from goal")
        axes[2].legend()
        # cost
        axes[3].plot(time_steps[:len(cost_ekf)], cost_ekf, label="cost_ekf", color='blue')
        axes[3].plot(time_steps[:len(cost_ukf)], cost_ukf, label="cost_ukf", color='orange')
        axes[3].grid(True)
        axes[3].set_xlabel("time")    
        axes[3].set_ylabel("cost-to-go")
        axes[3].legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir + f"performance_results-noise{noise}-dt{dt}.png")
        # plt.show()
        
        
 
 
if __name__ == '__main__':
    main()