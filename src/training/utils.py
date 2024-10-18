import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt

def load_label_names(file_path="../src/data/pastis_label_names.json"):
    with open(file_path, 'r') as f:
        label_names = json.load(f)
    return label_names.values()

#label_names = load_label_names('.data/pastis_label_names.json')

def validate_model(model, val_loader, criterion=nn.CrossEntropyLoss(), device="cuda", evaluate=False, evaluation_path=None):
    """
    Validate the model on the validation set.

    Args:
        model (torch.nn.Module): The trained model to be validated.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        criterion (torch.nn.Module, optional): The loss function used for validation.
            Defaults to nn.CrossEntropyLoss().
        device (torch.device, optional): The device on which the model and data are located.
            Defaults to "cuda".
        evaluate (bool, optional): Whether to evaluate the model's performance further. 
            Defaults to False.
        evaluation_path (str, optional): The path to save the evaluation results.
            Should be in the format "...<path to folder>/<experiment prefix>_{}",
            where '{}' will be replaced with the ending and file extension.
            e.g. "/mnt/c/repos/am4cm/evals/best_model_LTAE_{}"

    Returns:
        tuple: A tuple containing the validation loss, the targets, and the predictions made by the model.

    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    targets = []
    predictions = []

    with torch.no_grad():
        for inputs, target in val_loader:
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            total += target.size(0)
            correct += (torch.argmax(outputs, dim=1) == target).sum().item()
            if evaluate:
                predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.append(target.cpu().numpy())

    
    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy:.2f}%")

    if evaluate:
        targets = np.concatenate(targets)
        predictions = np.concatenate(predictions)

        label_names = load_label_names()
        report = classification_report(targets, predictions, target_names=label_names)
        # save classification report
        with open(evaluation_path.format('classification_report.txt'), 'w') as f:
            f.write(report)

        # confusion matrix, both true and pred normalized
        for normalize in [None, 'true', 'pred']:
            fig, ax = plt.subplots(figsize=(12, 12))
            cm = confusion_matrix(targets, predictions, normalize=normalize)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
            disp.plot(include_values=True, cmap='Greens', ax=ax, xticks_rotation='vertical')
            disp.ax_.set_title(f'Confusion matrix, normalize={normalize}')
            disp.ax_.figure.savefig(evaluation_path.format(f'confusion_matrix_{normalize}.png'))

    if evaluate:
        return val_loss, targets, predictions
    else:
        return val_loss
