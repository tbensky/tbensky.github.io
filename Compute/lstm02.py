import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math
import time
import os
import random
import sqlite3
import datetime

database_file = "../../220722.sqlite"

class neural_net(nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()

        # the 512 and 2 layers were chosen by trial and error, but seem to work ok
        c = 512
        self.lstm = nn.LSTM(100, c, num_layers=2, batch_first=True)

        self.relu = nn.Tanh() #nn.ReLU()
        self.fc1 = nn.Linear(c, 100)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.fc1(x)
        return x
        

def count_params(model):
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def find_speed(): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu") and torch.backends.mps.is_available():
       device = torch.device("mps")
    return device


#
# gets data in form [[100D t0],[100D t1],[100D t2]...[100D tn]]
#
def get_predict_data():
    times = []
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute("select distinct time from fish order by time")
    rows = cursor.fetchall()
    for row in rows:
        times.append(float(row[0]))

    print("Compiling data...")
    data = []
    for time in times:
        vec = []
        cursor.execute(f"select x,y,vx,vy from fish where time={time} order by fish")
        rows = cursor.fetchall()
        for row in rows:
            for e in row:
                vec.append(float(e))
        data.append(vec)
    return data


def normalize(v): 
    return (v - mean) / std

def unnormalize(v): 
    return v * std + mean

#
# start here
#

data = get_predict_data()
print(f"data length={len(data)}")
print("t,x,y,vx,vy")
t = 0
for d in data:
    for f in d:
        print(f"{t},{f[0]},{f[1]},{f[2]},{f[3]},{f[4]}\n")
exit()


#steps to train on
steps = 100

#these will be the training sequences
#"multi-step sequence learning"
#
# creates pairs like these:
# input: 0 to 100, target: 1 to 101
# input: 1 to 101, target: 2 to 102
# input: 2 to 102, target: 3 to 103
# input: 3 to 103, target: 4 to 104
# input: 4 to 104, target: 5 to 105
# input: 5 to 105, target: 6 to 106
# input: 6 to 106, target: 7 to 107
# input: 7 to 107, target: 8 to 108
# input: 8 to 108, target: 9 to 109
# input: 9 to 109, target: 10 to 110
# input: 10 to 110, target: 11 to 111
# input: 11 to 111, target: 12 to 112
# input: 12 to 112, target: 13 to 113
# input: 13 to 113, target: 14 to 114
# input: 14 to 114, target: 15 to 115
# input: 15 to 115, target: 16 to 116
# input: 16 to 116, target: 17 to 117
# input: 17 to 117, target: 18 to 118
#
# each data[ ] appended to input_seqs or target_seqs is an array of time-frames
#
input_seqs = []
target_seqs = []
for i in range(len(data)-steps):
    input_seqs.append(  data[   i       :   i+steps])           #append [i][i+1][i+2]...[i+steps]
    target_seqs.append( data[   i + 1   :   i + steps + 1])     #append [i+1][i+2][i+3]...[i+steps+1]

print(f"input_seqs size={len(input_seqs)}")

input_seqs = torch.tensor(input_seqs,dtype=torch.float32)
target_seqs = torch.tensor(target_seqs,dtype=torch.float32)


#get normalization params
all_data = torch.tensor(data,dtype=torch.float32)
mean = all_data.mean(dim=0)
std = all_data.std(dim=0)

input_seqs = normalize(input_seqs)
target_seqs = normalize(target_seqs)

train_dataset = TensorDataset(input_seqs[:1000],target_seqs[:1000])
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

device = find_speed()
print(f"GPU/CPU={device}")

ann = neural_net()
ann.to(device)
print(f"model parameters={count_params(ann):,}")

optimizer = optim.Adam(ann.parameters(),lr=1e-3)
loss_fn = nn.MSELoss()
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

os.system("rm loss.csv")

es = time.time()
start_time = es
epoch = 0
loss_track = []


ann.train()
while True:
    total_loss = 0.0
    es = time.time()

    for input_seq, target_seq in train_loader:
        input_seq = input_seq.to(device)        # (batch,step,size) so (32,100,100)
        target_seq = target_seq.to(device)      # also (32,100,100)

        if ann.training:
            noise = 0.01 * torch.randn_like(input_seq)
            input_seq = input_seq + noise
    
        # Predict the entire sequence at once
        pred_seq = ann(input_seq)               # shape: (32, 100, 100)

        loss = loss_fn(pred_seq, target_seq)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        #scheduler.step()
        #torch.nn.utils.clip_grad_norm_(ann.parameters(), max_norm=1.0)
        optimizer.zero_grad()

    des = time.time() - es
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss}, epoch time={des:.2f} s, total time={(time.time()-start_time)/60:.2f} min")
        loss_track.append([epoch,total_loss])
        x, y = zip(*loss_track)
        plt.plot(x,y)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("loss.png",dpi=300)
        plt.close()

        torch.save(ann.state_dict(), 'model_weights.pth')

        ann.eval()
        results = []
        input_t = normalize(torch.tensor(data[0])).to(device) #(100,)

        with torch.no_grad():
            for step in range(steps):

                #input_t is (100,)
                #need to add the batch size and other dimension for the LSTM
                
                #delta = ann(input_t.unsqueeze(0).unsqueeze(0))          # input_t: (100,) -> (1, 1, 100). delta: (1,1,100)
                #pred_next = input_t.unsqueeze(0).unsqueeze(0) + delta    # (1, 1, 100) + (1,1, 100)
                
                pred_next = ann(input_t.unsqueeze(0).unsqueeze(0))  # predicts absolute state directly

                #get rid of the (1, ) dimension
                pred_list = unnormalize(pred_next.squeeze(0).squeeze(0).cpu()).tolist()
                data_list = data[step]
                
                results.append([
                                [pred_list[0],data_list[0]],
                                [pred_list[4],data_list[4]],
                                [pred_list[8],data_list[8]],
                                [pred_list[12],data_list[12]]
                                ]
                                )
                input_t = pred_next.detach().squeeze(0).squeeze(0)    #input_t back to (100)

            N = 4
            plt.subplot(N,1,1)

            for n in range(N):
                plt.subplot(N,1,n+1)
                nn = []
                data_t = []
                for quad in results:
                    nn.append(quad[n][0])
                    data_t.append(quad[n][1])
                plt.plot(data_t, '.',label='Data')
                plt.plot(nn, '-',label='NN')
                plt.title(f"x of Fish #{n}")
                if n == N-1:
                    plt.xlabel("Time step")
                    plt.ylabel("x")
                    plt.legend()
                plt.tight_layout()
            plt.savefig("fish0.png",dpi=300)
            plt.close()
        ann.train()
    epoch += 1
    if total_loss < 0.001:
        break