import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
import nni

from resnet18 import ResNet, BasicBlock
from training_utils import train, validate
from utils import save_plots, get_data


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
plot_name = 'resnet_scratch'
print(model)
def func():
    global model, plot_name, device
    # Set seed.
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

    # Learning and training parameters.
    epochs = 20
    batch_size = 64
    learning_rate = 0.01
    


    # Define model based on the argument parser string.
    print('[INFO]: Training ResNet18 built from scratch...')
    

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    train_loader, valid_loader = get_data(batch_size=batch_size)
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, 
            valid_loader, 
            criterion,
            device
        )
        nni.report_intermediate_result(valid_epoch_acc)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
    
    nni.report_final_result(valid_acc)
    # Save the loss and accuracy plots.
    save_plots(
        train_acc, 
        valid_acc, 
        train_loss, 
        valid_loss, 
        name=plot_name
    )
    print('TRAINING COMPLETE')
    
from nni.nas.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(func)
import nni.nas.strategy as strategy
search_strategy = strategy.Random()
from nni.nas.experiment import NasExperiment
exp = NasExperiment(model, evaluator, search_strategy)
exp.config.max_trial_number = 3   # spawn 3 trials at most
exp.config.trial_concurrency = 1  # will run 1 trial concurrently
exp.config.trial_gpu_number = 1
exp.config.training_service.use_active_gpu = True
exp.run(port=8081)
for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)
