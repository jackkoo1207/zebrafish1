import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from local_param import DiscreteImage,Move,angle_between_2vec
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self,in_channels,num_channel=4, num_classes=6):
        super(ResNet18, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(num_channel, in_channels, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, in_channels, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 2*in_channels, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4*in_channels, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 8*in_channels, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8*in_channels, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
def train_model(model, INPUT1,INPUT2,OUTPUT, NUM_BATCH,DATA_SIZE,batch_size,epochs,grid,r,R,device='cuda'):
    best_test_loss = float('inf')
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # Loss function and optimizer
    # Training loop
    train_losses=[]
    test_losses=[]
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_loop = tqdm(range(NUM_BATCH), desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for _ in train_loop:
            train_loop.update(1)
            rand_index=np.random.randint(low=0,high=DATA_SIZE-1,size=batch_size)
            input1=INPUT1[rand_index]
            input2=INPUT2[rand_index]
            labels=OUTPUT[rand_index]
            input_data=np.zeros((batch_size,4,grid,grid))
            for i in range(batch_size):
                input_data[i] = DiscreteImage(input2[i,:,0],input2[i,:,1],input2[i,:,2],input1[i,0],input1[i,1],input2[i,:,3],input2[i,:,4],input2[i,:,5].astype(bool),input2[i,:,6],grid=grid,R=R,r=r)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(torch.from_numpy(input_data).float().to(device))
            loss = criterion(outputs,torch.from_numpy(labels).float().to(device))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            # Update progress bar
            train_loop.set_postfix(loss=loss.item())
        
        # Calculate training metrics
        train_loss /=(NUM_BATCH)
        print(f'Epoch {epoch+1}/{epochs} [Train] Loss: {train_loss:.4f}')
        train_losses.append(train_loss )
    print('Training complete!')
    return model,train_losses
def test_model(model,initial_midline,Test_frame,DATA_circle,R,grid,bool_have_wall):
    NUM_FISH=initial_midline.shape[0]
    new_midline=np.zeros((Test_frame,NUM_FISH, 3, 2))
    new_Observed_index=np.zeros((Test_frame-1,NUM_FISH, NUM_FISH))
    new_Observed_distance=np.zeros((Test_frame-1,NUM_FISH, NUM_FISH))
    new_Observed_phi=np.zeros((Test_frame-1,NUM_FISH, NUM_FISH))
    new_Observed_head_phi=np.zeros((Test_frame-1,NUM_FISH, NUM_FISH))

    new_L1=np.zeros((Test_frame-1, NUM_FISH))
    new_L2=np.zeros((Test_frame-1, NUM_FISH))
    new_theta=np.zeros((Test_frame-1, NUM_FISH))
    new_Radius=np.zeros((Test_frame-1, NUM_FISH))
    new_global_head_phi=np.zeros((Test_frame-1, NUM_FISH))
    new_midline[0]=initial_midline
    Predict_output=np.zeros((Test_frame, 2, 6))
    r=DATA_circle[2]
    with torch.no_grad():
        for i in tqdm(range(Test_frame-1)):
            new_head                =new_midline[i,:,0,:]-new_midline[i,:,1,:]
            new_tail                =new_midline[i,:,2,:]-new_midline[i,:,1,:]
            new_position            =new_midline[i,:,1,:]-DATA_circle[None,:2]
            new_global_head_phi[i]  =angle_between_2vec(new_head,-new_position, axis=1)
            new_theta[i]            =angle_between_2vec(-new_head, new_tail, axis=1)
            new_L1[i]               =np.linalg.norm(new_head, axis=1)
            new_L2[i]               =np.linalg.norm(new_tail, axis=1)
            new_Observed_vector     =new_midline[i,None,:,1,:]-new_midline[i,:,None,1,:]
            new_Observed_distance[i]=np.linalg.norm(new_Observed_vector, axis=2)
            new_Observed_phi[i]     =angle_between_2vec(new_head[:,None,:], new_Observed_vector, axis=2)
            new_Observed_head_phi[i]=angle_between_2vec(new_Observed_vector,new_head[None,:,:], axis=2)
            new_Observed_index[i]   =new_Observed_distance[i]<=R
            if bool_have_wall:
                new_Radius[i]=r-np.linalg.norm(new_position, axis=1)
                new_Radius[i,new_Radius[i]>R]=-1
            else:
                new_Radius[i]=-1
            input_data              = np.zeros((NUM_FISH,4,grid,grid))
            for fish_id in range(NUM_FISH):
                input_data[fish_id] = DiscreteImage(Observed_index=new_Observed_index[i,fish_id],Observed_distance=new_Observed_distance[i,fish_id],Observed_phi=new_Observed_phi[i,fish_id],L1=new_L1[i],L2=new_L2[i],theta=new_theta[i],Radius=new_Radius[i,fish_id],global_head_phi=new_global_head_phi[i,fish_id],Observed_head_phi=new_Observed_head_phi[i,fish_id],grid=grid,r=r,R=R)
            input_data = torch.tensor(input_data, dtype=torch.float32).to('cuda')
            Predict_output[i] = model(input_data).detach().cpu().numpy()
            new_midline[i+1]=Move(new_midline[i],Predict_output[i,:,0],Predict_output[i,:,1],Predict_output[i,:,2], Predict_output[i,:,3:])
    return new_midline, new_L1,new_L2,new_theta,new_global_head_phi,new_Radius,new_Observed_distance,new_Observed_phi,new_Observed_index,new_Observed_head_phi
