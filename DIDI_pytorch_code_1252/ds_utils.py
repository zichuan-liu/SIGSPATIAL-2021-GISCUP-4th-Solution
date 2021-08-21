from torch.utils.data import *
import msgpack
import threading

class GisDS(Dataset):
    def __init__(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            self.data = msgpack.unpackb(data, use_list=False)
        except:
            self.data = []
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

    
def get_train_dl(train_ds, num_workers=0, pin_memory=False, 
                 collate_fn=None,
                 batch_size=3):
    
    train_dl = DataLoader(train_ds,
                          collate_fn=collate_fn, 
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
#                           drop_last=False,
                          drop_last=True,
                          shuffle=True)
    return train_dl

def get_valid_dl(valid_ds, num_workers=0, pin_memory=False, 
                 collate_fn=None,
                 batch_size=3):
    
    val_dl = DataLoader(valid_ds, 
                        collate_fn=collate_fn, 
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
#                         drop_last=False,
                        drop_last=True,
                        shuffle=False)
    return val_dl

class DsLoader():
    def __init__(self, cache_all=False):
        self.ds_dct = {}
        self.cache_all = cache_all
        
    def get_train_ds(self, day):
        if self.cache_all==False:
            # del prev day
            pre_key = f"{day-1:02}"
            if pre_key in self.ds_dct:
                del self.ds_dct[pre_key]
            
        key = f"{day:02}"
        if key not in self.ds_dct:
            ds = self.load_ds(day)
            self.ds_dct[key] = ds
            self.preload_next(day+1)
            
            return ds
        else:
            ds = self.ds_dct[key]
            self.preload_next(day+1)
            return ds
        
    def preload_next(self, day):
        threading.Thread(
                    target=self.preload_ds, args=(day,), 
                    daemon=True
            ).start()
    
    def preload_ds(self, day):
        key = f"{day:02}"
        if key not in self.ds_dct:
            self.ds_dct[key]=GisDS(f"./data/new_msgpack/202008{day:02}.msgpack")
        
    def load_ds(self, day):
        return GisDS(f"./data/new_msgpack/202008{day:02}.msgpack")