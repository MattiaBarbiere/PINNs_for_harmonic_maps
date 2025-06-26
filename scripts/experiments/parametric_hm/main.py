"""
Parametric Harmonic Map Training Script.

This script trains a parametric harmonic map neural network model and saves
the training results including model weights and loss histories.

The script uses a specialized training function for parametric models that
can handle PDE parameters as additional network inputs.
"""

import torch

from ml_utils import *
from model import ParametricHmModel

# Initialize the parametric harmonic map model
model = ParametricHmModel()

# Train the model with specified hyperparameters
errors, grad_errors, loss, BC_loss = train_parametric_hm(
    model, 
    n_epochs=15000,
    batch_size=32,
    optimizer_threshold=8000,
    optimizer="Adam",
    loss_BC_weight=20,
    save_BC_loss=True
)

# Save the trained model state
torch.save(model.state_dict(), "model.pt")

# Save training history for analysis and plotting
torch.save(errors, "errors.pt")
torch.save(grad_errors, "grad_errors.pt")
torch.save(loss, "loss.pt")

# Save boundary condition loss if available
if BC_loss is not None:
    torch.save(BC_loss, "BC_loss.pt")