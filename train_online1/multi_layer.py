import torch
from torch import nn, optim
import numpy as np
import memtorch
from memtorch.utils import LoadMNIST
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128)
parser.add_argument('-e','--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('-hd','--hidden_size', type=int, default=10)

class MRFCLayer(object):
    def __init__(self, in_channels=28, out_channels=10, R_on=1600, R_off=0.1,
                 input_Rs=np.inf, time_step=1e-10, charge_steps=(6, 3),
                 Vp=20, Vn=-0.3, thres_Vc=1e-10):
        super().__init__()
        reference_memristor = memtorch.bh.memristor.VTEAM
        self.circuit_array = [[reference_memristor(time_series_resolution=time_step,
                                                   r_on=R_on, r_off=R_off)
                               for _ in range(in_channels + 1)]
                              for _ in range(out_channels)]  # Extra one for bias
        for i, single_layer in enumerate(self.circuit_array):
            for j, mem_ristor in enumerate(single_layer):
                rand_charge = np.random.randint(15,25)
                mem_ristor.simulate(torch.FloatTensor([Vp] * rand_charge))

        self.shape = (out_channels, in_channels + 1)
        self.input_Rs = input_Rs
        self.G = self.get_conductance_matrix() - 1 / input_Rs
        self.time_step = time_step
        self.charge_steps = charge_steps
        self.Vp = Vp
        self.Vn = Vn
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

    def update(self):
        grad1 = self.neural.weight.grad
        grad2 = self.neural.bias.grad
        grad = torch.concat((grad1,grad2.unsqueeze(-1)),dim=-1)
        
        # delta_W = np.outer(delta_V, np.append(inputs, 1))
        for i, single_layer in enumerate(self.circuit_array):
            for j, mem_ristor in enumerate(single_layer):
                if grad[i, j] > self.thres:
                    log_grad = np.log10(grad[i, j].numpy())   #(-10,2) -> (1,3) x/6+8/3
                    charge_step = int(log_grad/6 + 8/3)
                    mem_ristor.simulate(torch.FloatTensor([self.Vn] * charge_step))
                elif grad[i, j] < -self.thres:
                    log_grad = np.log10(-grad[i, j].numpy())   #(-10,0) -> (1,21) 2x+21
                    charge_step = int(2*log_grad + 21)
                    mem_ristor.simulate(torch.FloatTensor([self.Vp] * charge_step))
        self.G = self.get_conductance_matrix() - 1 / self.input_Rs
        self.neural.weight = torch.nn.Parameter(torch.tensor(self.G[:,:-1],dtype=torch.float32))
        self.neural.bias = torch.nn.Parameter(torch.tensor(self.G[:,-1],dtype=torch.float32))

class MultiMRFC(object):
    def __init__(self,in_channels,hidden_size,out_channels):
        super().__init__()
        self.layer1 = MRFCLayer(in_channels,hidden_size)
        self.layer2 = MRFCLayer(hidden_size,out_channels)
        self.relu = nn.LeakyReLU(0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor):
        neural_out = self.layer1(inputs)
        neural_out = self.relu(neural_out)
        neural_out = self.layer2(neural_out)
        neural_out = self.softmax(neural_out)
        return neural_out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)    
    
    def update(self, criterion, optimizer, preds: torch.Tensor, targets: torch.Tensor):
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.layer1.update()
        self.layer2.update()

        return loss

@torch.no_grad()
def test(model, test_loader):
    total = 0
    correct = 0
    for i, (data,target) in enumerate(test_loader):
        data = data.view(-1,784)
        neural_out = model(data)
        pred = torch.argmax(neural_out,dim=1)
        correct += torch.sum(pred == target).item()
        total += len(target)
    print('Test acc:',correct/total)

if __name__ == '__main__':
    args = parser.parse_args()
    train_loader, _, test_loader = LoadMNIST(batch_size=args.batch_size, validation=False)

    model = MultiMRFC(784, args.hidden_size, 10)
    optimizer = optim.SGD(itertools.chain(model.layer1.neural.parameters(), model.layer2.neural.parameters()), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs
    print('test before training...')
    test(model, test_loader)
    print('start training...')
    for e in range(epochs):
        total = 0
        correct = 0
        loss_log = []
        for i, (data,target) in enumerate(train_loader):
            data = data.view(-1,784)
            neural_out = model(data)
            pred = torch.argmax(neural_out,dim=1)
            correct += torch.sum(pred == target).item()
            total += len(target)
            loss = model.update(criterion, optimizer, neural_out, target)
            loss_log.append(loss.item())
        print('Train acc:',correct/total)
        print('loss:',np.mean(np.array(loss_log)))
        test(model, test_loader)