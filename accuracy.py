from sklearn.metrics import classification_report
import torch

def calc_accuracy(model, dataloader, class_names):
    y_true, y_pred = [], []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
