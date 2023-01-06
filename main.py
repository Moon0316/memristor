import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import memtorch
from memtorch.utils import LoadMNIST
from memtorch.mn.Module import patch_model
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
from model import minist, cifar10
from plot import visualize
from tqdm import tqdm
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--train",action='store_true',help='whether to train a DNN model')
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--step_size", type=int, default=2)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--nonideal",type=str,default=None,help='Add non-ideal characteristics: device|endurance|retention|finite|nonlinear')
parser.add_argument("--dataset", type=str, default='minist', help='type of dataset: minist|cifar10')

args = parser.parse_args()

device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')

def test(model,test_loader):
    correct = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):        
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()

    return 100. * float(correct) / float(len(test_loader.dataset))

def train(args, epoch, model, optimizer, scheduler, criterion, train_loader, test_loader):    
    best_acc = 0
    for i in range(epoch):
        model.train()
        e = i+1
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        for i, (input,target) in pbar:
            optimizer.zero_grad()
            input = input.to(device)
            output = model(input)
            loss = criterion(output,target.to(device))
            loss.backward()
            optimizer.step()
            pbar.set_description('EPOCH:{}, LOSS:{}, LR:{}'.format(e,loss,optimizer.param_groups[0]['lr']))
        scheduler.step()
        acc = test(model,test_loader)
        print('Test Accuracy: %2.2f%%' % acc)
        if acc > best_acc:
            best_acc = acc
            print('EPOCH:{}, save best model!'.format(e))
            torch.save(model.state_dict(), '{}_best_model.pt'.format(args.dataset))          
         
def MDNN(model,nonideal=None):
    reference_memristor = memtorch.bh.memristor.VTEAM
    reference_memristor_params = {'time_series_resolution': 1e-10}
    visualize(reference_memristor,reference_memristor_params)
    patched_model = patch_model(copy.deepcopy(model),
                          memristor_model=reference_memristor,
                          memristor_model_params=reference_memristor_params,
                          module_parameters_to_patch=[torch.nn.Conv2d],
                          mapping_routine=naive_map,
                          transistor=True,
                          programming_routine=None,
                          tile_shape=(128, 128),
                          max_input_voltage=0.3,
                          scaling_routine=naive_scale,
                          ADC_resolution=8,
                          ADC_overflow_rate=0.,
                          quant_method='linear')
    patched_model.tune_()
    
    if nonideal=='device':
        print('Consider DeviceFaults.')
        patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                        non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
                                        lrs_proportion=0.25,
                                        hrs_proportion=0.10,
                                        electroform_proportion=0)
    
    elif nonideal=='endurance':
        print('Consider Endurance.')
        patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.Endurance],
                                  x=1e4,
                                  endurance_model=memtorch.bh.nonideality.endurance_retention_models.model_endurance_retention,
                                  endurance_model_kwargs={
                                        "operation_mode": memtorch.bh.nonideality.endurance_retention_models.OperationMode.sudden,
                                        "p_lrs": [1, 0, 0, 0],
                                        "stable_resistance_lrs": 100,
                                        "p_hrs": [1, 0, 0, 0],
                                        "stable_resistance_hrs": 1000,
                                        "cell_size": 10,
                                        "temperature": 350,
                                  })
    
    elif nonideal=='retention':
        print('Consider Retention.')
        patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.Retention],
                                  time=1e3,
                                  retention_model=memtorch.bh.nonideality.endurance_retention_models.model_conductance_drift,
                                  retention_model_kwargs={
                                        "initial_time": 1,
                                        "drift_coefficient": 0.1,
                                  })
        
    elif nonideal=='finite':
        print('Consider FiniteConductanceStates.')
        patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.FiniteConductanceStates],
                                  conductance_states=5)  
        
    elif nonideal=='nonlinear':
        print('Consider NonLinear.')
        patched_model = apply_nonidealities(copy.deepcopy(patched_model),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.NonLinear],
                                  simulate=True)          
    
    else:
        print('Not Consider the nonideal characteristics')
        
    return patched_model


def main(args):
    if args.dataset == 'minist':
        model = minist().to(device)
    elif args.dataset == 'cifar10':
        model = cifar10().to(device)
    
    batch_size=args.batch_size
    
    if args.dataset=='minist':
        train_loader, _, test_loader = LoadMNIST(batch_size=batch_size, validation=False)
    elif args.dataset=='cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    if args.train:
        # hyper parameters
        lr = args.lr
        step_size= args.step_size
        epoch= args.epoch     

        # training preparation  
        optimizer = optim.Adam(model.parameters(),lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.2)
        criterion = nn.CrossEntropyLoss()
        
        # train
        train(args, epoch, model, optimizer, scheduler, criterion, train_loader, test_loader)
    
    # load best model
    model.load_state_dict(torch.load('{}_best_model.pt'.format(args.dataset)))
    acc=test(model,test_loader)
    print('DNN Test Accuracy: %2.2f%%' % acc)
    
    # conversion to MDNN
    patched_model = MDNN(model,args.nonideal)
    acc=test(patched_model,test_loader)
    print('MDNN Test Accuracy: %2.2f%%' % acc)
    

if __name__ == '__main__':
    main(args)