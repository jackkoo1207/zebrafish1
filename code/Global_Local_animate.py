import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig, [ax1,ax2] = plt.subplots(nrows=1, ncols=2)
def Global_Local_animate(midline,Observed_distance,Observed_phi,L1,L2,theta,Radius,global_head_phi,Observed_head_phi,Observer_index=0,t_start=0,Frame_interval=100):
    def animate(i):
        i+=t_start
        ax1.clear()
        ax2.clear()
        ax1.plot(midline[i,:,0,0], midline[i,:,0,1], 'o', color='red', markersize=1)
        ax1.plot(midline[i,:,1,0], midline[i,:,1,1], 'o', color='blue', markersize=1)
        ax1.plot(midline[i,:,2,0], midline[i,:,2,1], 'o', color='green', markersize=1)
        circle = plt.Circle((pos[i,Observer_index,0], pos[i,Observer_index,1]), R, color='black', fill=False)
        ax1.add_artist(circle)
        ax1.set_xlim(0, vid.shape[1])
        ax1.set_ylim(0, vid.shape[0])
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title(f"Global: Frame {i}")
        circle=plt.Circle((x, y), r, color='r',fill=False)
        ax1.add_patch(circle)

        Each_obs_index=Observed_index[i,Observer_index]
        CM_x=Observed_distance[i,Observer_index,Each_obs_index]*np.cos(Observed_phi[i,Observer_index,Each_obs_index])
        CM_y=Observed_distance[i,Observer_index,Each_obs_index]*np.sin(Observed_phi[i,Observer_index,Each_obs_index])
        ax2.plot(L1[i,Each_obs_index]*np.cos(Observed_phi[i,Observer_index,Each_obs_index]+Observed_head_phi[i,Observer_index,Each_obs_index])+CM_x, \
                L1[i,Each_obs_index]*np.sin(Observed_phi[i,Observer_index,Each_obs_index]+Observed_head_phi[i,Observer_index,Each_obs_index])+CM_y, \
                    'o', color='red', markersize=1)
        ax2.plot(CM_x,CM_y, \
                    'o', color='blue', markersize=1)
        ax2.plot(L2[i,Each_obs_index]*np.cos(Observed_phi[i,Observer_index,Each_obs_index]+Observed_head_phi[i,Observer_index,Each_obs_index]+np.pi+theta[i,Each_obs_index])+CM_x, \
                L2[i,Each_obs_index]*np.sin(Observed_phi[i,Observer_index,Each_obs_index]+Observed_head_phi[i,Observer_index,Each_obs_index]+np.pi+theta[i,Each_obs_index])+CM_y, 'o', color='green', markersize=1)
        circle = plt.Circle((0, 0), R, color='black', fill=False)
        ax2.add_artist(circle)
        ax2.set_xlim(-R, R)
        ax2.set_ylim(-R, R)
        ax2.set_title(f"Local: Frame {i}")
        ax2.set_aspect('equal', adjustable='box')
        if Radius[i,Observer_index]!=-1:
            circle=plt.Circle(((r-Radius[i,Observer_index])*np.cos(-global_head_phi[i,Observer_index]), -(r-Radius[i,Observer_index])*np.sin(-global_head_phi[i,Observer_index])), r, color='r',fill=False)
        ax2.add_patch(circle)
    return FuncAnimation(fig, animate, frames=Frame_interval, repeat=False)