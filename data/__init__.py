import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class ImageFolderWithPaths(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        for cls in self.class_names:
            cls_dir = os.path.join(root, cls)
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.samples.append((os.path.join(cls_dir, img), self.class_to_idx[cls]))
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

def get_dataloaders(base_dir, batch_size=32):
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    train_ds = ImageFolderWithPaths(train_dir)
    val_ds = ImageFolderWithPaths(val_dir)
    test_ds = ImageFolderWithPaths(test_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader, train_ds.class_names
