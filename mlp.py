from torch import nn 
import torch 

class MLP(nn.Module):
    def __init__(self,emb_size=16):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Linear(in_features=1*28*28,out_features=emb_size),
            nn.ReLU(),
            nn.Linear(in_features=emb_size,out_features=10),
        )
        
    def forward(self,x): # (batch_size,channel=1,width=28,height=28)
        return self.seq(x.view(x.size(0),-1))
    
if __name__=='__main__':
    mlp=MLP()
    x=torch.rand(5,1,28,28)
    y=mlp(x)
    print(y.shape)