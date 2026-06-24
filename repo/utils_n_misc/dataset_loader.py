from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import DataLoader
import time

st = time.time()

def LoadData(name = "ogbl-collab"):
    dataset = PygLinkPropPredDataset(name) 

LoadData()

et = time.time() - st
print(f"Done in {et} seconds")