from sklearn.metrics import precision_recall_curve
import torch
import torchvision
import numpy as np
import torch.nn as nn
import tensorboard_helper
import pytorch_dataset
import train_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import average_precision_score


device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def iou_pairs(orig_label, pc):

  rem_label = []
  for i in range(orig_label.shape[0]):
    flag = 0
    for j in range(orig_label.shape[0]):
      if i!=j:
        xA = max(orig_label[i,1], orig_label[j,1])
        yA = max(orig_label[i,2], orig_label[j,2])
        xB = min(orig_label[i,3], orig_label[j,3])
        yB = min(orig_label[i,4], orig_label[j,4])

        inter = max(0, xB - xA) * max(0, yB - yA)

        boxAarea = (orig_label[i,3] - orig_label[i,1]) * (orig_label[i,4] - orig_label[i,2])
        boxBarea = (orig_label[j,3] - orig_label[j,1]) * (orig_label[j,4] - orig_label[j,2])

        ou = inter / float(boxAarea + boxBarea - inter)

        if ou >= 0.5 and pc[i] < pc[j]:
          flag = 1

    if flag == 0:
      rem_label.append(orig_label[i])

  rem_label = torch.stack(rem_label)
  return rem_label

def mat_to_label(gt_matrix, threshold = 0.6):
    labels = np.zeros((0, 9))
    for i in range(gt_matrix.shape[1]):
      for j in range(gt_matrix.shape[2]):
        if gt_matrix[0, i, j] >= threshold:
          class_number = np.argmax([gt_matrix[5, i, j], gt_matrix[6, i, j], gt_matrix[7, i, j]])
          x1 = float(j*16 + gt_matrix[1, i, j] - gt_matrix[3, i, j]*64)
          y1 = float(i*16 + gt_matrix[2, i, j] - gt_matrix[4, i, j]*64)
          x2 = float(x1 + gt_matrix[3, i, j]*128)
          y2 = float(y1 + gt_matrix[4, i, j]*128)
          po = gt_matrix[0,i,j].detach().cpu().numpy()
          pp = gt_matrix[5,i,j].detach().cpu().numpy()
          pt = gt_matrix[6,i,j].detach().cpu().numpy()
          pc = gt_matrix[7,i,j].detach().cpu().numpy()
          labels = np.vstack((labels, np.array([class_number, x1, y1, x2, y2, po, pp, pt, pc])))
    return torch.from_numpy(labels).float()

def label_to_mat(label):
    output_label = np.zeros((8,8,8))
    center_x = (label[:,1] + label[:,3])/2
    center_y = (label[:,2] + label[:,4])/2
    x_indices = (center_x/16).astype(int)
    y_indices = (center_y/16).astype(int)
    output_label[0,y_indices,x_indices] = label[:,5]
    output_label[1,y_indices,x_indices] = center_x%16
    output_label[2,y_indices,x_indices] = center_y%16
    output_label[3,y_indices,x_indices] = abs(label[:,3] - label[:,1])/128.0
    output_label[4,y_indices,x_indices] = abs(label[:,4] - label[:,2])/128.0
    output_label[5,y_indices,x_indices] = label[:,6]
    output_label[6,y_indices,x_indices] = label[:,7]
    output_label[7,y_indices,x_indices] = label[:,8]
    return torch.from_numpy(output_label).double()

def plot_curve():
    train_loader, test_loader = pytorch_dataset.get_loaders()
    model = train_model.YOLOish().to(device).double()
    # Download the model from the link
    # https://drive.google.com/file/d/1Oed91n7DKPdR0PwX_Y_hL0UGJMzYPO-P/view?usp=sharing
    checkpoint = torch.load('yolo_model.pt')
    model.load_state_dict(checkpoint['net'])
    y_true = np.zeros((0, 3))
    y_score = np.zeros((0, 3))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device).double()
            labels = labels.to(device).double()
            outputs = model(images)
            # No post processing
            fin_output = []
            for j in range(outputs.shape[0]):
              np_pred_label = mat_to_label(outputs[j,:,:,:],0.0)
              # Remove Pr < 0.6
              pred_label = mat_to_label(outputs[j,:,:,:], 0.6)
              orig_label = mat_to_label(labels[j,:,:,:], 1.0)
              # NMS
              if pred_label.shape[0] == 0:
                fin_output.append(torch.zeros(8,8,8).double())
                continue
              rem_label = iou_pairs(pred_label, pred_label[:,5])
              fin_output.append(label_to_mat(rem_label.numpy()))

            fin_output = torch.stack(fin_output)
            y_score_batch = np.zeros((fin_output.shape[0]*8*8, 3))
            #print(((outputs[:, 0, :, :] > 0.6).float()*outputs[:, 5, :, :]).flatten())
            y_score_batch[:, 0] = ((fin_output[:, 0, :, :] > 0.6).double()*fin_output[:, 5, :, :]).flatten().detach().cpu().numpy()
            y_score_batch[:, 1] = ((fin_output[:, 0, :, :] > 0.6).double()*fin_output[:, 6, :, :]).flatten().detach().cpu().numpy()
            y_score_batch[:, 2] = ((fin_output[:, 0, :, :] > 0.6).double()*fin_output[:, 7, :, :]).flatten().detach().cpu().numpy()
            y_score = np.vstack((y_score, y_score_batch))

            y_true_batch = np.zeros((labels.shape[0]*8*8, 3))

            y_true_batch[:, 0] = labels[:, 5, :, :].flatten().detach().cpu().numpy()
            y_true_batch[:, 1] = labels[:, 6, :, :].flatten().detach().cpu().numpy()
            y_true_batch[:, 2] = labels[:, 7, :, :].flatten().detach().cpu().numpy()

            y_true = np.vstack((y_true, y_true_batch))

    classLabels = ['Pedestrian', 'Traffic Light', 'Car']
    mAPs = np.zeros(3)
    total = 0
    for cl in range(3):
        AP_cl = average_precision_score(y_true[:,cl], y_score[:,cl])
        mAPs[cl] = AP_cl*np.sum(y_true[:,cl])
        total+=np.sum(y_true[:,cl])
        print("Average Precision class {} is {}".format(cl,AP_cl))

    print(np.sum(mAPs)/total)
    for cl in range(3):
        precision, recall, threshold = precision_recall_curve(y_true[:,cl], y_score[:,cl])
        print(precision)
        print(recall)
        plt.plot(recall,precision,label=classLabels[cl])
        plt.title('Precision Recall curver for class {}'.format(cl))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    plot_curve()
