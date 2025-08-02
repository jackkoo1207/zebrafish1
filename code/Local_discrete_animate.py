import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig, [ax1,ax2] = plt.subplots(nrows=1, ncols=2)
def Global_Local_animate(midline,L1,L2,theta,Observed_distance,Observed_index,Observed_phi,Radius,r,R,Observer_index=0,t_start=0,Frame_interval=100):
    def animate(i):
        ax1.clear()
        ax2.clear()
        i+=t_start
        Each_obs_index=Observed_index[i,Observer_index]
        CM_x=Observed_distance[i,Observer_index,Each_obs_index]*np.cos(Observed_phi[i,Observer_index,Each_obs_index])
        CM_y=Observed_distance[i,Observer_index,Each_obs_index]*np.sin(Observed_phi[i,Observer_index,Each_obs_index])
        ax1.plot(L1[i,Each_obs_index]*np.cos(Observed_phi[i,Observer_index,Each_obs_index]+Observed_head_phi[i,Observer_index,Each_obs_index])+CM_x, \
                L1[i,Each_obs_index]*np.sin(Observed_phi[i,Observer_index,Each_obs_index]+Observed_head_phi[i,Observer_index,Each_obs_index])+CM_y, \
                    'o', color='red', markersize=1)
        ax1.plot(CM_x,CM_y, \
                    'o', color='blue', markersize=1)
        ax1.plot(L2[i,Each_obs_index]*np.cos(Observed_phi[i,Observer_index,Each_obs_index]+Observed_head_phi[i,Observer_index,Each_obs_index]+np.pi+theta[i,Each_obs_index])+CM_x, \
                L2[i,Each_obs_index]*np.sin(Observed_phi[i,Observer_index,Each_obs_index]+Observed_head_phi[i,Observer_index,Each_obs_index]+np.pi+theta[i,Each_obs_index])+CM_y, 'o', color='green', markersize=1)
        circle = plt.Circle((0, 0), R, color='black', fill=False)
        ax1.add_artist(circle)
        ax1.set_xlim(-R, R)
        ax1.set_ylim(-R, R)
        ax1.set_title(f"Local: Frame {i}")
        ax1.set_aspect('equal', adjustable='box')
        if Radius[i,Observer_index]!=-1:
            circle=plt.Circle(((r-Radius[i,Observer_index])*np.cos(-global_head_phi[i,Observer_index]), -(r-Radius[i,Observer_index])*np.sin(-global_head_phi[i,Observer_index])), r, color='r',fill=False)
        ax1.add_patch(circle)
        Image=DiscreteImage(i, Observer_index=Observer_index,Observed_index=Observed_index,Observed_distance=Observed_distance,Observed_phi=Observed_phi,L1=L1,L2=L2,theta=theta,Radius=Radius,global_head_phi=global_head_phi,Observed_head_phi=Observed_head_phi)
        ax2.imshow(np.sum(Image, axis=0))
        ax2.invert_yaxis()
    return FuncAnimation(fig, animate, frames=Frame_interval, repeat=False)