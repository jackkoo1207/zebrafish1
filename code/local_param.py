import numpy as np
R=300
grid=64
scale=(grid//2)/R
DATA_circle=np.load('data_circle/2.npy')
x,y,r=DATA_circle
def angle_between_2vec(vec1, vec2, axis=2):
    """Calculate the angle between two vectors."""
    return np.arctan2(
        np.cross(vec1, vec2, axis=axis),
        np.sum(vec1 * vec2, axis=axis)
    )
def local_param(midline,r,R):
    new_head                =new_midline[:,0,:]-new_midline[:,1,:]
    new_tail                =new_midline[:,2,:]-new_midline[:,1,:]
    new_position            =new_midline[:,1,:]-DATA_circle[None,:2]
    new_global_head_phi  =angle_between_2vec(new_head,-new_position, axis=1)
    new_theta            =angle_between_2vec(-new_head, new_tail, axis=1)
    new_L1               =np.linalg.norm(new_head, axis=1)
    new_L2               =np.linalg.norm(new_tail, axis=1)
    new_Observed_vector     =new_midline[None,:,1,:]-new_midline[:,None,1,:]
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
def DiscreteImage(L1,L2,theta,global_head_phi,Radius,Observed_distance,Observed_phi,Observed_index,Observed_head_phi):
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