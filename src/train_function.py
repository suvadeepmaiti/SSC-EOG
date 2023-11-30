
import time
import torch

def train(model, trainDataloader, criterion, optimizer, epoch):
    total_loss = 0.
    all_loss = 0.
    log_interval = 50
    start_time = time.time()
    num_batches = len(trainDataloader)
    
    for batch, train_data in enumerate(trainDataloader):
        data, tgt = train_data
        data, tgt = data.to(DEVICE), tgt.type(torch.LongTensor).to(DEVICE)
        
        pred = model(data)
        
        # Zero the gradients for every batch
        optimizer.zero_grad()
        
        # Prediction
        custom_pred = pred[:, int(pred.shape[1] / 2), :]
        
        # Compute the loss and its gradients
        loss = criterion(custom_pred, tgt)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Adjust learning weights
        optimizer.step()

        total_loss += loss.item()
        all_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'loss {cur_loss:5.4f}')
            total_loss = 0
            start_time = time.time()
            
    return all_loss / (batch + 1)
