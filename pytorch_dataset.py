# torch and torchvision imports
import torch
import torchvision
import numpy as np

class HW2Dataset(torch.utils.data.Dataset):
  def __init__(self, image, given_label,transform = None):

    self.image = image
    self.label = given_label
    self.transform = transform

  def __getitem__(self, index):
    output_label = np.zeros((8,8,8))
    center_x = (self.label[index][:,1] + self.label[index][:,3])/2
    center_y = (self.label[index][:,2] + self.label[index][:,4])/2
    x_indices = (center_x/16).astype(int)
    y_indices = (center_y/16).astype(int)
    output_label[0,y_indices,x_indices] = 1.0
    output_label[1,y_indices,x_indices] = center_x%16
    output_label[2,y_indices,x_indices] = center_y%16
    output_label[3,y_indices,x_indices] = abs(self.label[index][:,3] - self.label[index][:,1])/128.0
    output_label[4,y_indices,x_indices] = abs(self.label[index][:,4] - self.label[index][:,2])/128.0
    classes = 5 + self.label[index][:,0].astype(int)
    output_label[classes,y_indices,x_indices] = 1.0
#     print(output_label[0,:,:])
    if self.transform:
      image_as_tensor = self.transform(self.image[index,:,:,:])
    output_as_tensor = torch.from_numpy(output_label)
    #print(image_as_tensor.shape)
    #print(output_as_tensor.shape)
#     print(output_as_tensor[index, 0, :, :])
    return image_as_tensor,output_as_tensor

  def __len__(self):
    return len(self.data.index)


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
                                            torchvision.transforms.Normalize((0.5,), (0.5,))])

def get_loaders():
    images =  np.load('images.npz', allow_pickle = True)
    labels =  np.load('labels.npz', allow_pickle = True, encoding='latin1')
    images = images['arr_0']
    labels = labels['arr_0']
    dataset = HW2Dataset(images, labels, transform = transform)
    idx = np.arange(0,images.shape[0],1)
    np.random.shuffle(idx)
    train_idx = idx[:int(0.80*images.shape[0])]
    test_idx = idx[int(0.80*images.shape[0]):]
    train_data = torch.utils.data.Subset(dataset, train_idx)
    test_data  = torch.utils.data.Subset(dataset, test_idx)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    return train_loader, test_loader
