from vit_model import vit_base_patch16_224 as create_model
from crossattention import Multi_CrossAttention as mcx_layer
from cat  import CATLayer1 as in_cross_layer
from cat import reverse as rev
from transformer_text import Transformer as txt_trans
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from test1 import train_loader1,test_loader1
# from transformers import BertModel, BertTokenizer

# bert_model = BertModel.from_pretrained('bert-base-uncased') #path
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')




class Contrastive_loss(nn.Module):
    def __init__(self,tau):
        super(Contrastive_loss,self).__init__()
        self.tau=tau
        # self.z1=z1
        # self.z2=z2

    def sim(self,z1:torch.Tensor,z2:torch.Tensor):
        z1=F.normalize(z1)
        z2=F.normalize(z2)
        return torch.mm(z1,z2.t())
    

    def semi_loss(self,z1:torch.Tensor,z2:torch.Tensor):
        f=lambda x: torch.exp(x/self.tau)
        refl_sim=f(self.sim(z1,z2))
        return -torch.log(refl_sim.diag()/refl_sim.sum(1))
    
    def forward(self,z1:torch.Tensor,z2:torch.Tensor):
        l1=self.semi_loss(z1,z2)
        l2=self.semi_loss(z2,z1)
        sum=(l1+l2)*0.5
        # print(sum.shape)
        sum_loss=sum.mean()
        # print(f'sum_loss {sum_loss}')
        return sum_loss





class Model1(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads):
        super(Model1, self).__init__()
        self.i_model1 = create_model(num_classes=1000)
        self.i_model2 = mcx_layer(embed_dim, embed_dim, num_heads)

    def forward(self, imagedata):
        i_feature1, q = self.i_model1(imagedata)
        i_feature2, i_tokens = self.i_model2(i_feature1, q)
        # combined_features = torch.cat((i_feature1, i_feature2), dim=1)
        # output = self.output_layer(combined_features)
        return i_feature2, i_tokens

model2=txt_trans()



def train_one_epoch(model1, model2, train_loader, criterion, optimizer1, optimizer2, device):
    model1.train()
    model2.train()
    total_loss = 0.0
    
    for images, texts in train_loader:
        images = images.to(device)
        # for i in range(len(texts)):
        #     texts[i] = texts[i].to(device)
        texts=texts.to(device)

        
        # Forward pass through model1 (image model)
        i_feature2, i_tokens = model1(images)
        
        # Forward pass through model2 (text model)
        txt_feature, txt_tokens = model2(texts)
        txt_feature, txt_tokens=  txt_feature.to(device), txt_tokens.to(device)

        # Compute the contrastive loss between i_tokens and txt_tokens
        loss = criterion(i_tokens, txt_tokens)
        
        # Zero gradients, backward pass, and optimization step for model1
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        loss=loss.item()
        # total_loss += loss.item() * images.size(0)
        total_loss += loss
        print(f'batchloss is {loss}')
    epoch_loss = total_loss / len(train_loader.dataset)
    return epoch_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = Model1(num_classes=1000, embed_dim=768, num_heads=4).to(device)
model2 = txt_trans().to(device)

criterion = Contrastive_loss(0.07)  
optimizer1 = optim.Adam(model1.parameters(), lr=0.0001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.0001)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model1, model2, train_loader1, criterion, optimizer1, optimizer2, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")


    











# #structure of model s
#     i_model1=create_model(num_classes= )
#     #model.train()
#     i_feature1,q=i_model1(imagedata)
#     #i_feature,q:[B,N+1,embed_dim]
#     i_model2=mcx_layer(#embed_dim,embed_dim,num_heads)
#     i_feature2,i_tokens= i_model2(i_feature1,q)
#     #i_feature2:[B,N,embed_dim],i_tokens:[B,50(dim the same as txt_token)]
#     t_model1=txt_trans()
#     txt_feature,txt_token=t_model1(#txt) #txtsize=[B,padding_size,word_embedding_dim]
#     #txt_feature.size=txt.size,txt_token:[B,word_embedding_dim]
# #structure of model e