import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def example(rank, world_size):
    print("Starting")
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    print("Done")

def main():
    world_size = 2
    os.environ['MASTER_ADDR'] ='127.0.0.1' #'145.101.32.61'  # lisa IP
    os.environ['MASTER_PORT'] = '9028' #29500
    print("Spawning")
    mp.spawn(example,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__=="__main__":
    main()