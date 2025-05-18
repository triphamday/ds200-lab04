from pytorch_lightning import Trainer

def train_model(model, train_loader, val_loader, max_epochs=5):
    trainer = Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_loader, val_loader)
    return trainer
