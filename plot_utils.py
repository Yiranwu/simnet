import matplotlib.pyplot as plt
#from models import GCN2
import torch
import numpy as np

#model = GCN2(32, 6).double()
#model.load_state_dict(torch.load('gcn-2-60.pth'))
#model.cuda()
#model.eval()


def plot_channel_loss(f):
    l0,l1,l2,l3,l4,l5=[],[],[],[],[],[]
    t=[]
    tick=0
    plt.xlabel('epoch')
    plt.ylabel('loss')
    with open(f, 'rb') as file:
        for line in file:
            if line.startswith(b'loss'):
                print(line[19:])
                numbers=line[19:]
                numbers = [float(s) for s in numbers.split()]
                l0.append(numbers[0])
                l1.append(numbers[1])
                l2.append(numbers[2])
                l3.append(numbers[3])
                l4.append(numbers[4])
                l5.append(numbers[5])
                t.append([tick])
                tick+=1
        print(l0)
        print(l4)
        l0=np.array(l0)
        l1=np.array(l1)
        l2=np.array(l2)
        l3=np.array(l3)
        l4=np.array(l4)
        l5=np.array(l5)
        plt.plot(t,l0,label='angle')
        plt.plot(t,l1,label='px')
        plt.plot(t,l2,label='py')
        plt.plot(t,l3,label='vx')
        plt.plot(t,l4,label='vy')
        plt.plot(t,l5,label='w')
        plt.plot(t,l0+l1+l2+l3+l4+l5, label='loss')
        plt.legend()
        plt.show()

def plot_loss_curve(f):
    train, test = [], []
    t=[]
    tick=0
    plt.xlabel('epoch')
    plt.ylabel('loss')
    with open(f, 'rb') as file:
        for line in file:
            if line.startswith(b'train'):
                numbers = float(line[15:])
                train.append(numbers)
                t.append([tick])
            if line.startswith(b'test'):
                numbers = float(line[15:])
                test.append(numbers)
                tick+=1
        print(train)
        print(test)
        train=np.array(train)
        test=np.array(test)
        plt.plot(t,train,label='train')
        plt.plot(t,test,label='test')
        plt.legend()
        plt.show()


#plot_channel_loss('within_log.txt')
plot_loss_curve('train_eval.txt')