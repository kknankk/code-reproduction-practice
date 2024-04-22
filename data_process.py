import pandas as pd
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def load_dicts():
    train_qa_path = './pvqa/qas/train/train_qa.pkl'
    with open(train_qa_path, 'rb') as f:
        train_dict = pickle.load(f)
    test_qa_path = './pvqa/qas/test/test_qa.pkl'
    with open(test_qa_path, 'rb') as f:
        test_dict = pickle.load(f)
    val_qa_path = './pvqa/qas/val/val_qa.pkl'
    with open(val_qa_path, 'rb') as f:
        val_dict = pickle.load(f)
    return train_dict,test_dict,val_dict

def concat_qa(dict):
    img_id = []
    caption = []
    for item in dict:
        img_id.append(item['image'])
        caption.append(item['question']+' '+ item['answer'])
    df_dict = {'img_id':img_id,'caption':caption}
    df = pd.DataFrame(df_dict)
    return df

if __name__ == '__main__':
    train_dict,test_dict,val_dict = load_dicts()
    train_df = concat_qa(train_dict)
    test_df = concat_qa(test_dict)
    val_df = concat_qa(val_dict)
    print(train_df.shape, test_df.shape, val_df.shape)
    train_df.to_csv('./pvqa/images/train.csv', index=False)
    test_df.to_csv('./pvqa/images/test.csv', index=False)
    val_df.to_csv('./pvqa/images/val.csv', index=False)
# print(caption)



