import numpy as np 
import torch as torch
from load_mnist import process_data
import argparse
import tqdm

parser = argparse.ArgumentParser(description='1T1R Training')

parser.add_argument('--lr', type=float,default=1e-5,help ='learning rate')
parser.add_argument('--vol_base', type=float,default=0.15)
parser.add_argument('--beta', type=float,default=1.5)
parser.add_argument('--lrs', type=float,default=4e-5,help ='Initialized weight:LRS')
parser.add_argument('--batch_size',type=int,default=100)
parser.add_argument('--right_activation_gt',type=float,default=1)
parser.add_argument('--wrong_activation_gt',type=float,default=0.)
parser.add_argument('--input_dim',type=int,default=784)
parser.add_argument('--output_dim',type=int,default=10)
parser.add_argument('--epoch_num',type=int,default=10)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--use_sign',action='store_true')
parser.add_argument('--reverse',action='store_true')
parser.add_argument('--add_noise',action='store_true')
parser.add_argument('--sigma',type=float,default=0.1,help='Sigma of noise')
args = parser.parse_args()


LR = args.lr
VOL = args.vol_base
BETA = args.beta
LRS = args.lrs
BATCH_SIZE = args.batch_size

T_LABEL = args.right_activation_gt
F_LABEL = args.wrong_activation_gt

INPUT_DIM = args.input_dim
OUT_DIM = args.output_dim

EPOCH_NUM = args.epoch_num
SEED = args.seed
REVERSE = args.reverse
USE_SIGN = args.use_sign

ADD_NOISE = args.add_noise
SIGMA = args.sigma


def process_input(image,label,reverse=False):
    '''
    Input:image (B,C)
    '''
    right_index = (label==1.)
    wrong_index = (label==0.)
    activation_gt = torch.zeros_like(label)
    activation_gt[right_index] = T_LABEL
    activation_gt[wrong_index] = F_LABEL

    if reverse:
        voltage = VOL*(255-image)
    else:
        voltage = VOL*image
    return voltage,activation_gt

def init_weight():
    weight = torch.ones(INPUT_DIM,OUT_DIM)*LRS
    return weight.double()

def update_weight(voltage,weight,activation_gt,use_sign=False,add_noise=False,sigma=0):

    current = torch.mm(voltage,weight) #(B,OUT_DIM)
    activation = torch.tanh(BETA*current) #(B,OUT_DIM)
    grad = (1-activation*activation)*BETA
    hidden = (activation_gt-activation)*grad #(B,OUT_DIM)
    delta = torch.bmm(voltage.unsqueeze(-1),hidden.unsqueeze(1)) #(B,INPUT,OUTPUT)
    if use_sign:
        delta_weight = torch.sign(delta.sum(dim=0))
    else:
        delta_weight = delta.sum(dim=0)
    
    if add_noise:
        noise = sigma*torch.randn(weight.shape)
        return weight + LR*(delta_weight + noise)
    return weight + LR*delta_weight
    
def inference(voltage,weight):
    current = torch.mm(voltage,weight) #(B,OUT_DIM)
    activation = torch.tanh(BETA*current) #(B,OUT_DIM)
    output = torch.argmax(activation,dim=-1)   
    return output

def cal_acc(output,label):    
    label_ = torch.argmax(label,dim=-1)
    return (label_ == output).sum()/output.shape[0]
    
if __name__ == '__main__':

    train_x,train_y = process_data("train")
    test_x,test_y = process_data("test")

    np.random.seed(SEED)
    np.random.shuffle(train_x)
    np.random.seed(SEED)
    np.random.shuffle(train_y)
    
    train_x = torch.from_numpy(train_x).double()
    train_y = torch.from_numpy(train_y).double()
    test_x = torch.from_numpy(test_x).double()
    test_y = torch.from_numpy(test_y).double()

    weight = init_weight()

    ITER_NUM = train_x.shape[0]//BATCH_SIZE

    for epoch in range(EPOCH_NUM):
        for iters in tqdm.tqdm(range(ITER_NUM)):
            train_voltage,train_activation_gt = process_input(train_x[iters*BATCH_SIZE:(iters+1)*BATCH_SIZE],train_y[iters*BATCH_SIZE:(iters+1)*BATCH_SIZE],reverse=REVERSE)
            weight = update_weight(train_voltage,weight,train_activation_gt,use_sign=USE_SIGN,add_noise=ADD_NOISE,sigma=SIGMA)
            if iters %10 ==0:
                test_voltage,test_activation_gt = process_input(test_x,test_y,reverse=REVERSE)
                output = inference(test_voltage,weight)
                Accuracy = cal_acc(output,test_y)
                print(f"Test Accuracy is {Accuracy}")
                
    torch.save(weight,'weight.pt')
    test_voltage,test_activation_gt = process_input(test_x,test_y,reverse=REVERSE)
    output = inference(test_voltage,weight)
    Accuracy = cal_acc(output,test_y)
    print(f"Final Test Accuracy is {Accuracy}")
    
    


        



