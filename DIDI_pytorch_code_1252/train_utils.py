from tqdm.notebook import tqdm, trange
import torch
import torch.nn.functional as F
import wandb
from IPython.core.debugger import set_trace
import os

eta_min, eta_max, eta_mean, eta_std = (2.3978952727983707, 9.371353167823885, 6.553886963677842, 0.5905307292899195)

def dict2device(data, device):
    if device.type!='cpu':
        for item in data:
            if torch.is_tensor(data[item]):
                data[item] = data[item].to(device)
        return data
    else:
        return data

scaler = torch.cuda.amp.GradScaler()

def train(epoch_idx, date, model, train_dl, optimizer, device, mixed=True):
    
    mae = 0
    mape = 0
    num_item = 0
    
    model.train()
    tqdm_bar = tqdm(train_dl, leave=False)
    for batch_idx, (data, target) in enumerate(tqdm_bar):
        tqdm_bar.set_description(f"Train ")
        data, target = dict2device(data, device), target.to(device)
        optimizer.zero_grad()
        
        if mixed==True:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = F.l1_loss(output, target)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = F.l1_loss(output, target)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            # epoch_loss+=loss
            y_pred = torch.exp(output*eta_std + eta_mean)
            y_true = torch.exp(target*eta_std + eta_mean)

            diff = torch.abs(y_pred - y_true)

            num_item += len(diff)

            mae += sum(diff)

            mape += sum(diff/y_true)
            
    mae = (mae/num_item).item()
    mape = (mape/num_item).item()

    wandb.log({f'Train_mae_08{date:02}': mae})

    wandb.log({f'Train_mape_08{date:02}': mape})
    
    
def valid(model, valid_dl, device, save_ckpt):
    
    mae = 0.
    mape = 0.
    num_item = 0.
    
    model.eval()
    with torch.no_grad():
        tqdm_bar = tqdm(valid_dl, leave=False)
        for batch_idx, (data, target) in enumerate(tqdm_bar):
            tqdm_bar.set_description(f"Valid ")
            data, target = dict2device(data, device), target.to(device)

            output = model(data)
            loss = F.l1_loss(output, target)

            y_pred = torch.exp(output*eta_std + eta_mean)
            y_true = torch.exp(target*eta_std + eta_mean)

            diff = torch.abs(y_pred - y_true)

            num_item += len(diff)
            
            mae += sum(diff)
            mape += sum(diff/y_true)
    
    mae = (mae/num_item).item()
    mape = (mape/num_item).item()
    
    if save_ckpt:
        os.makedirs('./ckpt', exist_ok=True)
        torch.save(model.state_dict(), f"./ckpt/model_mae_{mae:.5f}.pt")

    
    wandb.log({'Valid_mae_epoch': mae})
    wandb.log({'Valid_mape_epoch': mape})
    
    return mae, mape