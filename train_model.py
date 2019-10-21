import torch.nn as nn
import torch
import torchvision
import numpy as np
from sklearn.metrics import average_precision_score
import tensorboard_helper
import pytorch_dataset

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class YOLOish(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU())
    self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU())
    self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU())
    self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU())
    self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU())
    self.layer6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU())
    self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU())
    self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU())
    self.layer9 = nn.Conv2d(64, 8, kernel_size=3, stride = 1, padding = 1)
    self.sig = nn.Sigmoid()

  def forward(self, X):
    out = self.layer1(X)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.layer7(out)
    out = self.layer8(out)
    out = self.layer9(out)
    out[:,0,:,:] = self.sig(out[:,0,:,:])
    out[:,1,:,:] = self.sig(out[:,1,:,:])*16
    out[:,2,:,:] = self.sig(out[:,2,:,:])*16
    out[:,3,:,:] = self.sig(out[:,3,:,:])
    out[:,4,:,:] = self.sig(out[:,4,:,:])
    out[:,5,:,:] = self.sig(out[:,5,:,:])
    out[:,6,:,:] = self.sig(out[:,6,:,:])
    out[:,7,:,:] = self.sig(out[:,7,:,:])
    return out


def criterion(outputs, labels):
  l_coord = 5
  l_noobj = 0.5
  xy_loss = l_coord*torch.sum(labels[:,0,:,:]*((labels[:,1,:,:] - outputs[:,1,:,:])**2 + (labels[:,2,:,:] - outputs[:,2,:,:])**2))
  wh_loss = l_coord*torch.sum(labels[:,0,:,:]*((torch.sqrt(labels[:,3,:,:]) - torch.sqrt(outputs[:,3,:,:]))**2 + (torch.sqrt(labels[:,4,:,:]) - torch.sqrt(outputs[:,4,:,:]))**2))

  c_loss = torch.sum(labels[:,0,:,:]*((labels[:,0,:,:] - outputs[:,0,:,:])**2))
  noobj_loss = l_noobj*torch.sum((1-labels[:,0,:,:])*((labels[:,0,:,:] - outputs[:,0,:,:])**2))
  class_loss = torch.sum(labels[:,0,:,:]*((labels[:,5,:,:] - outputs[:,5,:,:])**2 + (labels[:,6,:,:] - outputs[:,6,:,:])**2 + (labels[:,7,:,:] - outputs[:,7,:,:])**2))
  return xy_loss + wh_loss + c_loss + noobj_loss + class_loss

def train(net, optimizer, criterion, train_loader, test_loader, epochs, model_name, plot):
    model = net.to(device)
    total_step = len(train_loader)
    overall_step = 0
    logger = tensorboard_helper.Logger('./logs')
    mAPs = np.zeros((epochs,3))
    for epoch in range(epochs):
        correct = 0
        total = 0
        y_true = np.zeros((0, 3))
        y_score = np.zeros((0, 3))
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to configured device
            images = images.to(device).double()
            labels = labels.to(device).double()
            #Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            overall_step+=1

            y_score_batch = np.zeros((outputs.shape[0]*8*8, 3))
            y_score_batch[:, 0] = ((outputs[:, 0, :, :] > 0.6).double()*outputs[:, 5, :, :]).flatten().detach().cpu().numpy()
            y_score_batch[:, 1] = ((outputs[:, 0, :, :] > 0.6).double()*outputs[:, 6, :, :]).flatten().detach().cpu().numpy()
            y_score_batch[:, 2] = ((outputs[:, 0, :, :] > 0.6).double()*outputs[:, 7, :, :]).flatten().detach().cpu().numpy()
            y_score = np.vstack((y_score, y_score_batch))

            y_true_batch = np.zeros((labels.shape[0]*8*8, 3))

            y_true_batch[:, 0] = labels[:, 5, :, :].flatten().detach().cpu().numpy()
            y_true_batch[:, 1] = labels[:, 6, :, :].flatten().detach().cpu().numpy()
            y_true_batch[:, 2] = labels[:, 7, :, :].flatten().detach().cpu().numpy()

            y_true = np.vstack((y_true, y_true_batch))

            if (i+1) % 10 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))
            if plot:
              info = { ('loss for' + model_name): loss.item()}

              for tag, value in info.items():
                logger.scalar_summary(tag, value, overall_step+1)
        total = 0
        for cl in range(3):
          AP_cl = average_precision_score(y_true[:,cl], y_score[:,cl])
          num_objs = AP_cl*np.sum(y_true[:,cl])
          mAPs[epoch,cl] = num_objs
          total += np.sum(y_true[:,cl])
          print("Average Precision for epoch {} class {} is {}".format(epoch, cl,AP_cl))
        info = { ('MAP ' + model_name): np.sum(mAPs[epoch])/total}
        for tag, value in info.items():
          logger.scalar_summary(tag, value, epoch+1)


def start_train():
    epochs = 30
    model = YOLOish().to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader = pytorch_dataset.get_loaders()
    train(model, optimizer, criterion, train_loader, test_loader, epochs, 'YOLO',True)
    checkpoint = {'net': model.state_dict()}
    torch.save(checkpoint,'yolo_model.pt')

if __name__ == '__main__':
    start_train()
