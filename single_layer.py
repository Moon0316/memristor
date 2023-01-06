import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import memtorch
from memtorch.utils import LoadMNIST


class MRFCLayer(object):
    def __init__(self, in_channels=28, out_channels=10, R_on=1600, R_off=0.1,
                 input_Rs=np.inf, time_step=2e-9, charge_steps=(4, 2),
                 Vc=1, thres_Vc=1e-5, use_activation=True):
        super().__init__()
        reference_memristor = memtorch.bh.memristor.VTEAM
        self.circuit_array = [[reference_memristor(time_series_resolution=time_step,
                                                   r_on=R_on, r_off=R_off)
                               for _ in range(in_channels + 1)]
                              for _ in range(out_channels)]  # Extra one for bias
        for i, single_layer in enumerate(self.circuit_array):
            for j, mem_ristor in enumerate(single_layer):
                rand_charge = np.random.randint(20,30)
                mem_ristor.simulate(torch.FloatTensor([Vc] * rand_charge))

        self.shape = (out_channels, in_channels + 1)
        self.input_Rs = input_Rs
        self.G = self.get_conductance_matrix() - 1 / input_Rs
        self.time_step = time_step
        self.charge_steps = charge_steps
        self.Vc = Vc
        self.thres = thres_Vc

        self.neural = nn.Linear(in_channels, out_channels)
        self.neural.weight = torch.nn.Parameter(torch.tensor(self.G[:,:-1],dtype=torch.float32))
        self.neural.bias = torch.nn.Parameter(torch.tensor(self.G[:,-1],dtype=torch.float32))

    def get_conductance_matrix(self):
        ret = []
        for single_fc_layer in self.circuit_array:
            for mem_ris in single_fc_layer:
                ret.append(mem_ris.g)

        return np.array(ret).reshape(self.shape)

    def forward(self, inputs: torch.Tensor):
        neural_out = self.neural(inputs)
        return neural_out  # For use of computing middle loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, criterion, optimizer, preds: torch.Tensor, targets: torch.Tensor):
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        grad1 = self.neural.weight.grad
        grad2 = self.neural.bias.grad
        grad = torch.concat((grad1,grad2.unsqueeze(-1)),dim=-1)
        
        # delta_W = np.outer(delta_V, np.append(inputs, 1))
        for i, single_layer in enumerate(self.circuit_array):
            for j, mem_ristor in enumerate(single_layer):
                if grad[i, j] > self.thres:
                    mem_ristor.simulate(torch.FloatTensor([-self.Vc] * self.charge_steps[0]))
                elif grad[i, j] < -self.thres:
                    mem_ristor.simulate(torch.FloatTensor([self.Vc] * self.charge_steps[1]))
        self.G = self.get_conductance_matrix() - 1 / self.input_Rs
        self.neural.weight = torch.nn.Parameter(torch.tensor(self.G[:,:-1],dtype=torch.float32))
        self.neural.bias = torch.nn.Parameter(torch.tensor(self.G[:,-1],dtype=torch.float32))
        return loss

class dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index][0], self.data_list[index][1]

    def __len__(self):
        return len(self.data_list)

def test(model, test_loader):
    total = 0
    correct = 0
    model.neural.eval()
    for i, (data,target) in enumerate(test_loader):
        data = data.view(-1,784)
        neural_out = model(data)
        neural_out = softmax(neural_out)
        pred = torch.argmax(neural_out,dim=1)
        correct += torch.sum(pred == target).item()
        total += len(target)
    print('Test acc:',correct/total)

if __name__ == '__main__':
    batch_size = 64
    train_, _, test_ = LoadMNIST(batch_size=2, validation=False)

    train_list = []
    test_list = []
    for i, (data,target) in enumerate(train_):
        for i in range(target.shape[0]):
            if target[i] in [0,1,2]:
                train_list.append((data[i],target[i]))
    train_loader = DataLoader(dataset(train_list), batch_size=batch_size, shuffle=True)
    
    for i, (data,target) in enumerate(test_):
        for i in range(target.shape[0]):
            if target[i] in [0,1,2]:
                test_list.append((data[i],target[i]))
    test_loader = DataLoader(dataset(test_list), batch_size=batch_size, shuffle=False)

    model = MRFCLayer(784,3)
    optimizer = optim.SGD(model.neural.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=-1)
    epochs = 10
    print('test before training...')
    test(model, test_loader)
    print('start training...')
    for e in range(epochs):
        total = 0
        correct = 0
        loss_log = []
        model.neural.train()
        for i, (data,target) in enumerate(train_loader):
            data = data.view(-1,784)
            neural_out = model(data)
            neural_out = softmax(neural_out)
            pred = torch.argmax(neural_out,dim=1)
            correct += torch.sum(pred == target).item()
            total += len(target)
            loss = model.update(criterion, optimizer, neural_out, target)
            loss_log.append(loss.item())
        print('Train acc:',correct/total)
        print('loss:',np.mean(np.array(loss_log)))
        test(model, test_loader)