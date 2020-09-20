from torch import nn, optim

from configuration import TrainingConfig
from models import OneChannelResnet
from trainer import get_dataloaders
from torch.utils.tensorboard import SummaryWriter

def run_tensorboard(config_file):
    configuration = TrainingConfig()
    configuration.parse_config(config_file)

    class_labels = configuration.class_labels
    device = "cpu"
    restart_checkpoint = configuration.restart_checkpoint
    chpt_folder = configuration.chpt_folder
    epochs = configuration.epochs
    batch_size = configuration.batch_size
    feat_folder = configuration.feature_folder
    folds = configuration.folds
    experiment_name = configuration.experiment_name
    train_dataloader = get_dataloaders(feat_folder, class_labels, folds, batch_size)

    print(f"Using device: {device}")
    model = OneChannelResnet(class_labels, pretrained=False)
    model.to(device)

    # criterion = nn.BCEWithLogitsLoss()  # Use this when there is no sigmoid before models output
    criterion = nn.BCELoss()
    print(f"Loss fnc: {criterion}")

    optimizer = optim.Adam(model.parameters())
    print(f"Loss fnc: {optimizer}")

    writer = SummaryWriter(f'runs/{experiment_name}')
    images, labels = next(iter(train_dataloader))
    images = images.unsqueeze(1)

    writer.add_graph(model, images)
    writer.close()

if __name__ == "__main__":
    run_tensorboard("configs/training/resnet_quint_notes_v0.4.json")

