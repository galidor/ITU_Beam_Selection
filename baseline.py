from models import LidarMarcus, Lidar3D
from torch import nn
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from torch import optim
from ignite.metrics import TopKCategoricalAccuracy
import torch.nn.functional as F


def beams_log_scale(y, thresholdBelowMax):
    y_shape = y.shape

    for i in range(0, y_shape[0]):
        thisOutputs = y[i, :]
        logOut = 20 * np.log10(thisOutputs + 1e-30)
        minValue = np.amax(logOut) - thresholdBelowMax
        zeroedValueIndices = logOut < minValue
        thisOutputs[zeroedValueIndices] = 0
        thisOutputs = thisOutputs / sum(thisOutputs)
        y[i, :] = thisOutputs

    return y

def get_beam_output(output_file):
    thresholdBelowMax = 6

    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    y = yMatrix.reshape(yMatrix.shape[0], num_classes)
    y = beams_log_scale(y, thresholdBelowMax)

    return y, num_classes


def evaluate(net, test_dataloader):
    with torch.no_grad():
        net.eval()
        top_1 = TopKCategoricalAccuracy(k=1)
        for i, data in enumerate(test_dataloader):
            lidar, beams = data
            lidar = lidar.cuda()
            beams = beams.cuda()
            preds = net(lidar)
            preds = F.softmax(preds, dim=1)
            # print(top_k_accuracy(preds, beams, 10))
            # exit()
            top_1.update((preds, torch.argmax(beams)))
        net.train()
        print(top_1.compute())



if __name__ == '__main__':
    lidar_data_train = np.load("./data/baseline_data/lidar_input/lidar_train.npz")['input']
    lidar_data_test = np.load("./data/baseline_data/lidar_input/lidar_validation.npz")['input']

    coord_data_train = np.load("./data/baseline_data/coord_input/coord_train.npz")['coordinates']
    coord_data_test = np.load("./data/baseline_data/coord_input/coord_validation.npz")['coordinates']

    beam_data_train = np.load("./data/baseline_data/beam_output/beams_output_train.npz")['output_classification']
    beam_data_test = np.load("./data/baseline_data/beam_output/beams_output_validation.npz")['output_classification']

    beam_output_train = get_beam_output("./data/baseline_data/beam_output/beams_output_train.npz")
    beam_output_test = get_beam_output("./data/baseline_data/beam_output/beams_output_validation.npz")

    lidar_data_train = torch.from_numpy(lidar_data_train).float()
    lidar_data_test = torch.from_numpy(lidar_data_test).float()

    beam_output_train = torch.from_numpy(beam_output_train[0]).float()
    beam_output_test = torch.from_numpy(beam_output_test[0]).float()

    train_dataset = TensorDataset(lidar_data_train, beam_output_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(lidar_data_test, beam_output_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Lidar3D().cuda()
    optimizer = optim.Adam(model.parameters())
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    criterion = lambda y_pred, y_true: -torch.sum(torch.mean(y_true * torch.log(y_pred + 1e-30), axis=0))

    # evaluate(model, test_dataloader)
    top_1 = TopKCategoricalAccuracy(k=1)
    for i in range(100):
        accumulated_loss = []
        tbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in tbar:
            optimizer.zero_grad()
            lidar, beams = data
            lidar = lidar.cuda()
            beams = beams.cuda()
            preds = model(lidar)
            loss = criterion(F.softmax(preds, dim=1), beams)
            # top_1.update((F.softmax(preds), torch.argmax(beams)))
            loss.backward()
            optimizer.step()
            accumulated_loss.append(loss.item())
            tbar.set_postfix_str(str(sum(accumulated_loss)/len(accumulated_loss))[:5])
        # top_1.reset()
        evaluate(model, test_dataloader)
        model.train()
