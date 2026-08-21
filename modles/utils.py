from torch.utils.data import DataLoader

def prepare_dataloaders(dataset, receivers, batch_size=128):
    """Prepare dataloaders for training and evaluation.dataset is split into pretrain and train sets."""
    pretrain_loader = {}
    all_loader = {}
    for receiver in receivers:
        pretrain_size = int(len(dataset[receiver]) * 0.2)
        train_size = len(dataset[receiver]) - pretrain_size
        pretrain_dataset, train_dataset = dataset[receiver].split(pretrain_size, train_size)
        pretrain_loader[receiver] = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
        all_loader[receiver] = DataLoader(dataset[receiver], batch_size=batch_size, shuffle=True)
    return pretrain_loader, all_loader


