from data.data_loader import get_dataloaders
from model.CNN import SimpleCNN
from model.trainer import train_model
from accuracy import calc_accuracy

if __name__ == "__main__":
    data_split_dir = r"D:\ds200-lab04\output\parquet"
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_split_dir)
    model = SimpleCNN(num_classes=len(class_names))
    trainer = train_model(model, train_loader, val_loader, max_epochs=10)
    calc_accuracy(model, test_loader, class_names)
