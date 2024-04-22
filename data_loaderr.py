import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import nltk
import json
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

with open(r'E:\CODE\VIT\train_vocab.json','r') as f:
    data_vocab=json.load(f)
    vocab=data_vocab['word2idx']
    print(len(vocab))

unknown_token_index = len(vocab)+1


class ImageTextPairDataset(Dataset):
    def __init__(self, csv_file, image_transform=None, text_transform=None,split='train',vocab=vocab):
        self.data = pd.read_csv(csv_file)   #read_csv(path)
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.split=split
        self.vocab=vocab
        self.embedding=nn.Embedding(2850,50)
        self.rnn=nn.RNN(50,50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx,split='train'):
        
        img_id = self.data.iloc[idx]['img_id']
        img_path=os.path.join('pvqa\images',split,img_id+'.jpg')
        txt = self.data.iloc[idx]['caption']
        tokens=nltk.tokenize.word_tokenize(
            str(txt).lower()
        )
        captions=[]
        captions.append(vocab['<start>'])
        captions.extend([vocab[token] if token in vocab else unknown_token_index for token in tokens])
        # lst_int = list(map(int, lst))
        captions.append(vocab['<end>'])
        # print(captions)
        txt=torch.tensor(captions)

        txt=self.embedding(txt)




        # Load image
        img = Image.open(img_path).convert('RGB')
        # img = Image.open(img_path)
        
        # Apply image transformations
        if self.image_transform:
            img = self.image_transform(img)

        # Apply text transformations
        if self.text_transform:
            txt = self.text_transform(txt)


        return img, txt


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a fixed size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])



train_csv_path=r"E:\CODE\VIT\pvqa\images\train.csv"
test_csv_path=r"E:\CODE\VIT\pvqa\images\test.csv"
# Instantiate training and testing datasets with both image and text transformations
train_dataset = ImageTextPairDataset(train_csv_path, image_transform=image_transform, text_transform=None,split='train')
test_dataset = ImageTextPairDataset(test_csv_path, image_transform=image_transform, text_transform=None,split='test')


# print(train_dataset[0])

train_loader1 = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader1 = DataLoader(test_dataset, batch_size=64, shuffle=False)


  # Define padding function to ensure the same length of txt tensors within each batch
def custom_collate_fn(batch):
    images, texts = zip(*batch)
    # Pad the txt tensors within each batch
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    target_length = 196
    pad_length = target_length - padded_texts.size(1)
    if pad_length > 0:
        padding = torch.zeros((padded_texts.size(0), pad_length, padded_texts.size(2)), dtype=torch.long)
        padded_texts = torch.cat([padded_texts, padding], dim=1)
    return torch.stack(images), padded_texts




    return torch.stack(images), padded_texts

train_loader1.collate_fn = custom_collate_fn
test_loader1.collate_fn = custom_collate_fn

# for idx,(image,txt) in enumerate(train_loader1):
#     print(idx,image.shape,txt.shape)