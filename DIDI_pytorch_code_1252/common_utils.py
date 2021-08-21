import pickle
from pathlib import Path

def load_pickle(file_path):
    with open(file_path,'rb') as f:
        content = pickle.load(f)
    return content

def dump_pickle(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
        
def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


