import numpy as np

def angle_between_2vec(vec1, vec2, axis=2):
    """Calculate the angle between two vectors."""
    return np.arctan2(
        np.cross(vec1, vec2, axis=axis),
        np.sum(vec1 * vec2, axis=axis)
    )
def local_param(midline,DATA_circle,R):
    r=DATA_circle[2]
    new_head                =midline[:,0,:]-midline[:,1,:]
    new_tail                =midline[:,2,:]-midline[:,1,:]
    new_position            =midline[:,1,:]-DATA_circle[None,:2]
    new_global_head_phi  =angle_between_2vec(new_head,-new_position, axis=1)
    new_theta            =angle_between_2vec(-new_head, new_tail, axis=1)
    new_L1               =np.linalg.norm(new_head, axis=1)
    new_L2               =np.linalg.norm(new_tail, axis=1)
    new_Observed_vector     =midline[None,:,1,:]-midline[:,None,1,:]
    new_Observed_distance=np.linalg.norm(new_Observed_vector, axis=2)
    new_Observed_phi     =angle_between_2vec(new_head[:,None,:], new_Observed_vector, axis=2)
    new_Observed_head_phi=angle_between_2vec(new_Observed_vector,new_head[None,:,:], axis=2)
    new_Observed_index   =new_Observed_distance<=R
    new_Radius=r-np.linalg.norm(new_position, axis=1)
    new_Radius[new_Radius>R]=-1
    return new_L1,new_L2,new_theta,new_global_head_phi,new_Radius,new_Observed_distance,new_Observed_phi,new_Observed_index,new_Observed_head_phi
def mid_point_circle_algorithm(radius):
    """
    使用中點圓算法生成圓的邊界點。
    radius: 圓的半徑
    回傳: 圓的邊界點列表 [(x, y), ...]
    """
    points = []
    x = radius
    y = 0
    d = 1 - radius  # 初始決策參數

    while x >= y:
        # 八分之一對稱性
        points.append((x, y))
        points.append((y, x))
        points.append((-x, y))
        points.append((-y, x))
        points.append((-x, -y))
        points.append((-y, -x))
        points.append((x, -y))
        points.append((y, -x))

        y += 1

        if d <= 0:
            d += 2 * y + 1  # 垂直方向移動
        else:
            x -= 1
            d += 2 * y - 2 * x + 1  # 水平方向移動

    return points
def DiscreteImage(L1,L2,theta,global_head_phi,Radius,Observed_distance,Observed_phi,Observed_index,Observed_head_phi,grid,R,r):
    scale=(grid//2)/R
    Each_obs_index=Observed_index
    CM_x=Observed_distance[Each_obs_index]*np.cos(Observed_phi[Each_obs_index])
    CM_y=Observed_distance[Each_obs_index]*np.sin(Observed_phi[Each_obs_index])
    head=np.array([L1[Each_obs_index]*np.cos(Observed_phi[Each_obs_index]+Observed_head_phi[Each_obs_index])+CM_x, \
            L1[Each_obs_index]*np.sin(Observed_phi[Each_obs_index]+Observed_head_phi[Each_obs_index])+CM_y]).T
    body=np.array([CM_x,CM_y]).T
    tail=np.array([L2[Each_obs_index]*np.cos(Observed_phi[Each_obs_index]+Observed_head_phi[Each_obs_index]+np.pi+theta[Each_obs_index])+CM_x, \
            L2[Each_obs_index]*np.sin(Observed_phi[Each_obs_index]+Observed_head_phi[Each_obs_index]+np.pi+theta[Each_obs_index])+CM_y]).T
    head=(head*scale).astype(int)
    body=(body*scale).astype(int)
    tail=(tail*scale).astype(int)
    if Radius!=-1:
        circle=mid_point_circle_algorithm(r*scale)
        center=np.array([(r-Radius)*np.cos(-global_head_phi), -(r-Radius)*np.sin(-global_head_phi)])
        circle=np.array(circle)+(center*scale)[None,:]
        circle=circle.astype(int)
    Image=np.zeros((4,grid, grid))
    for pixel in head:
        if pixel[0]+grid//2>=0 and pixel[1]+grid//2>=0 and pixel[0]+grid//2 < grid and pixel[1] +grid//2< grid and np.linalg.norm(pixel)<=grid//2:
            Image[0,pixel[1]+grid//2,pixel[0]+grid//2] = 1
    for pixel in body:
        if pixel[0]+grid//2>=0 and pixel[1]+grid//2>=0 and pixel[0]+grid//2 < grid and pixel[1]+grid//2 < grid and np.linalg.norm(pixel)<=grid//2:
            Image[1,pixel[1]+grid//2,pixel[0]+grid//2] = 1
    for pixel in tail:
        if pixel[0]+grid//2>=0 and pixel[1]+grid//2>=0 and pixel[0]+grid//2 < grid and pixel[1]+grid//2 < grid and np.linalg.norm(pixel)<=grid//2:
            Image[2,pixel[1]+grid//2, pixel[0]+grid//2] = 1
    if Radius!=-1:
        for pixel in circle:
            if pixel[0]+grid//2>=0 and pixel[1]+grid//2>=0 and pixel[0]+grid//2 < grid and pixel[1]+grid//2 < grid and np.linalg.norm(pixel)<=grid//2:
                Image[3,pixel[1]+grid//2, pixel[0]+grid//2] = 1
    return Image
def Move(midline,L1_dot,L2_dot,theta_dot,OUTPUT):
    head=midline[:,0,:]-midline[:,1,:]
    tail=midline[:,2,:]-midline[:,1,:]
    theta=angle_between_2vec(-head, tail, axis=1)
    L1=np.linalg.norm(head, axis=1)
    L2=np.linalg.norm(tail, axis=1)
    v_para=OUTPUT[:,0]
    v_norm=OUTPUT[:,1]
    angular_speed=OUTPUT[:,2]
    head_phi=np.arctan2(head[:,1], head[:,0])
    new_head_phi=head_phi+angular_speed
    new_L1=L1+L1_dot
    new_L2=L2+L2_dot
    new_theta=theta+theta_dot
    new_head=new_L1[:,None]*(np.array([np.cos(new_head_phi), np.sin(new_head_phi)]).T)
    new_tail=new_L2[:,None]*(np.array([np.cos(new_head_phi+np.pi+new_theta), np.sin(new_head_phi+np.pi+new_theta)]).T)
    para=np.array([np.cos(head_phi), np.sin(head_phi)]).T
    Norm=np.array([-np.sin(head_phi), np.cos(head_phi)]).T
    new_velocity=v_para[:,None]*para+v_norm[:,None]*Norm
    body=midline[:,1,:]
    new_body=body + new_velocity
    new_midline=np.zeros((body.shape[0], 3, 2))
    new_midline[:,0]=new_head+new_body
    new_midline[:,1]=new_body
    new_midline[:,2]=new_tail+new_body
    return new_midline
def get_input_output(midline,DATA_circle,R):
    NUM_FISH=midline.shape[1]
    r=DATA_circle[2]
    head=midline[:,:,0,:]-midline[:,:,1,:]
    tail=midline[:,:,2,:]-midline[:,:,1,:]
    def angle_between_2vec(vec1, vec2, axis=2):
        """Calculate the angle between two vectors."""
        return np.arctan2(
            np.cross(vec1, vec2, axis=axis),
            np.sum(vec1 * vec2, axis=axis)
        )
    position=midline[:,:,1,:]-DATA_circle[None,None,:2]
    global_head_phi=angle_between_2vec(head,-position, axis=2)
    theta=angle_between_2vec(-head, tail, axis=2)
    L1=np.linalg.norm(head, axis=2)
    L2=np.linalg.norm(tail, axis=2)
    Observed_vector=midline[:,None,:,1,:]-midline[:,:,None,1,:]
    Observed_distance=np.linalg.norm(Observed_vector, axis=3)
    Observed_phi=angle_between_2vec(head[:,:,None,:], Observed_vector, axis=3)
    Observed_head_phi=angle_between_2vec(Observed_vector,head[:,None,:,:], axis=3)
    Observed_index=Observed_distance<=R
    Radius=r-np.linalg.norm(position, axis=2)
    Radius[Radius>R]=-1
    angular_speed= angle_between_2vec(head[:-1,:,:],head[1:,:,:], axis=2)
    velocity =midline[1:,:,1,:]-midline[:-1,:,1,:]
    speed=np.linalg.norm(velocity, axis=2)
    velocity_phi=angle_between_2vec(head[:-1,:,:],velocity,axis=2)
    v_para=speed*np.cos(velocity_phi)
    v_norm=speed*np.sin(velocity_phi)
    L1_dot=L1[1:]-L1[:-1]
    L2_dot=L2[1:]-L2[:-1]
    theta_dot=theta[1:]-theta[:-1]
    INPUT1=np.stack([global_head_phi[:-1],Radius[:-1]],axis=2)
    INPUT2=np.stack([np.tile(L1[:-1,None],(1,NUM_FISH,1)),np.tile(L2[:-1,None],(1,NUM_FISH,1)),np.tile(theta[:-1,None],(1,NUM_FISH,1)),Observed_distance[:-1],Observed_phi[:-1],Observed_index[:-1],Observed_head_phi[:-1]],axis=3)
    OUTPUT=np.stack([L1_dot,L2_dot,theta_dot,v_para,v_norm,angular_speed],axis=2)
    INPUT1=INPUT1.reshape(-1,2)
    INPUT2=INPUT2.reshape(-1,NUM_FISH,7)
    OUTPUT=OUTPUT.reshape(-1,6)
    return INPUT1, INPUT2, OUTPUT