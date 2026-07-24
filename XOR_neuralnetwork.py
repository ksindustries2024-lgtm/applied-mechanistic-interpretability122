#this code includes a logit gate type called XOR WHERE same input gives 0,different input gives 1 so it includes many nueral network fundamentals
#this is my first self build nueral network
import torch
import torch.nn as nn
X=torch.tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
y=torch.tensor([[0.0],[1.0],[1.0],[0.0]])

class SimpleNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.first_layer=nn.Linear(2,8)
    self.relu=nn.ReLU()
    self.second_layer=nn.Linear(8,1)
   
  def forward(self,x):
    x=self.first_layer(x)
    x=self.relu(x)
    x=self.second_layer(x)
    return torch.sigmoid(x)

model=SimpleNet()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
loss_fn=nn.BCELoss()

for epoch in range(1000):
  pred=model(X)
  loss=loss_fn(pred,y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if epoch%200==0:
    print(f"EPOCH:{epoch},loss:{loss.item()}")

with torch.no_grad():
  for i in range(4):
    print(f"{X[i].tolist()}->{pred[i].item():.3f},(target:{y[i].item()})")    



