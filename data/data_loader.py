# data/dataloader.py
import os
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, parquet_dir, transform=None):
        # Lấy các file bắt đầu bằng "part-" và có đuôi ".parquet" (dù phía trước là .snappy hay không)
        files = [
            os.path.join(parquet_dir, f)
            for f in os.listdir(parquet_dir)
            if f.startswith('part-') and f.endswith('.parquet')
        ]
        print("Files parquet:", files)
        assert len(files) > 0, f"Không tìm thấy file parquet trong {parquet_dir}"

        # Đọc tất cả file parquet vào DataFrame
        self.df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        self.classes = sorted(self.df['label'].unique())
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert('RGB')
        x = self.transform(img)
        y = self.cls2idx[row.label]
        return x, y

def get_dataloaders(parquet_dir, batch_size=32, transform=None):
    ds = ImageDataset(parquet_dir, transform)
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader, ds.classes


