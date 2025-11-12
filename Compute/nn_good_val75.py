import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

class ffnn(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=2304):
        super(ffnn, self).__init__()
        
        self.model = nn.Sequential(
                nn.Linear(4, 256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
            
                nn.Linear(1024, 2304)
            )

    def forward(self, x):
        return self.model(x)


class BinaryDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.fromfile(file_path, dtype=np.float32)
        sample_size = 4 + 2304
        assert len(self.data) % sample_size == 0, "Binary file size mismatch."
        self.samples = self.data.reshape(-1, sample_size)

        # get inputs and outputs
        self.inputs = self.samples[:,:4]
        self.outputs = self.samples[:,4:]

        # inputs
        self.inputs_mean = self.inputs.mean(axis=0)
        self.inputs_std = self.inputs.std(axis=0)
        self.inputs_std[self.inputs_std == 0] = 1.0  # avoid divide-by-zero

        # outputs
        self.outputs_mean = self.outputs.mean(axis=0)
        self.outputs_std = self.outputs.std(axis=0)
        self.outputs_std[self.outputs_std == 0] = 1.0

        # normalize per feature
        self.inputs = (self.inputs - self.inputs_mean) / self.inputs_std
        self.outputs = (self.outputs - self.outputs_mean) / self.outputs_std


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i = torch.tensor(self.inputs[idx], dtype=torch.float32)
        o = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return i, o

    def getio(self,idx):
        i = torch.tensor(self.inputs[idx], dtype=torch.float32)
        o = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return i, o

    def unnormalize_input(self, input_tensor):
        return input_tensor * torch.tensor(self.inputs_std, dtype=torch.float32, device=input_tensor.device) + torch.tensor(self.inputs_mean, dtype=torch.float32, device=input_tensor.device)


    def unnormalize_output(self, output_tensor):
        return output_tensor * torch.tensor(self.outputs_std, dtype=torch.float32, device=output_tensor.device) + torch.tensor(self.outputs_mean, dtype=torch.float32, device=output_tensor.device)


def count_params(model):
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def find_speed(): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu") and torch.backends.mps.is_available():
       device = torch.device("mps")
    return device


device = find_speed()
#device = torch.device('cpu')
#print("**USING CPU**")
print(f"GPU/CPU={device}")
os.system("rm Evolve/*.png")


# Load dataset
print("Loading data...")
dataset = BinaryDataset('data.bin')
i,o = dataset.getio(1)
test_vec = i.to(device)
print(test_vec)
print("Done loading data...")


dataset_size = len(dataset)
val_ratio = 0.2
val_size = int(dataset_size * val_ratio)
train_size = dataset_size - val_size

generator = torch.Generator().manual_seed(42)  # for reproducibility
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

#train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# i,o = dataset[0]
# print(i)
# img = o.reshape(32,72)
# plt.imshow(img, cmap='viridis', aspect='equal',interpolation='nearest',extent = [-0.45,3.05,0.02,0.62])
# plt.show()
# exit()

# Initialize model
model = ffnn().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)


loss_fn = nn.SmoothL1Loss() #nn.MSELoss()  # or nn.CrossEntropyLoss() depending on the task


# Training loop
loss_track = []
val_track = []
lr_history = []
epoch = 0
t0 = time.time()
frame_count = 0
while True:
    batch_t0 = time.time()
    loss_total = 0.0

    # batch_x, batch_y = dataset[0]
    # batch_x = batch_x.unsqueeze(0)

    # batch_x = batch_x.to(device)
    # batch_y = batch_y.to(device)
    
    # optimizer.zero_grad()
    # outputs = model(batch_x)
    # loss = loss_fn(outputs, batch_y)
    # loss_total += loss.item()
    # loss.backward()
    # optimizer.step()

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)


        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)
        loss_total += loss.item()
        loss.backward()
        optimizer.step()


    loss_track.append([epoch,loss_total])

    
    if epoch == 0 or epoch % 10 == 0:
        now = time.time()
        model.eval()

        val_correct = 0
        val_total = 0
        threshold = 0.01
        val_loss_total = 0.0

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                pred = model(val_x)

                # Compare predictions to ground truth
                diff = torch.abs(pred - val_y)

                # Option 1: Count output elements below threshold
                correct_per_element = (diff < threshold).sum().item()
                total_elements = torch.numel(diff)

                val_correct += correct_per_element
                val_total += total_elements
                val_loss = loss_fn(pred, val_y)
                val_loss_total += val_loss.item()
        
        val_loss_avg = val_loss_total / len(val_loader)
        scheduler.step(val_loss_avg)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append([epoch,current_lr])
        
        pv = val_correct / val_total * 100.0
        val_track.append([epoch,pv])
       
        print(f"Epoch {epoch}, Loss: {loss_total}, validation={pv:0.1f}%, val_loss_avg={val_loss_avg:.2f} {scheduler.get_last_lr()} lr={current_lr:.2e}, batch={now-batch_t0:.2f}s, total={(now-t0)/3600:.2f}h")
        
        x, y = zip(*loss_track)
        plt.subplot(3,1,1)
        plt.plot(x,y)
        #plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Veritas template NN training {(now-t0)/3600:.2f}h ")

        plt.subplot(3,1,2)
        x, y = zip(*val_track)
        plt.plot(x,y)
        #plt.xlabel("Epoch")
        plt.ylabel("Percent recognized")


        plt.subplot(3,1,3)
        x, y = zip(*lr_history)
        plt.plot(x,y)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.savefig("loss.png",dpi=300)
        plt.close()

       
        batch_x, batch_y = dataset.getio(0)
        batch_x = batch_x.unsqueeze(0).to(device)
        #see what batch_y looks like
        img = dataset.unnormalize_output(batch_y.cpu()).reshape(32,72)

        plt.subplot(2,1,1)
        plt.imshow(img, cmap='viridis', aspect='equal',interpolation='nearest',extent = [-0.45,3.05,0.02,0.62])
        plt.title("Data")
        
        plt.subplot(2,1,2)
        out = model(batch_x)
        out = out.detach().cpu()
        out = dataset.unnormalize_output(out)
        img = out.reshape(32,72)
        plt.imshow(img, cmap='viridis', aspect='equal',interpolation='nearest',extent = [-0.45,3.05,0.02,0.62])
        params = dataset.unnormalize_input(batch_x.detach())
        params = [round(x,2) for x in params.tolist()[0]]
        plt.title(rf"NN predict [z,$\lambda$,logE,b]={params}")
        plt.savefig(f"Evolve/evolve_{frame_count:05d}.png",dpi=300)
        plt.close()
        frame_count += 1
        model.train()

    epoch += 1