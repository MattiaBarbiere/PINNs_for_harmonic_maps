#hydra imports
import hydra
from omegaconf import DictConfig

# project imports
from hmpinn.models import ModelV2
from hmpinn.ML_utils import train

# poisson problem import
from hmpinn.PDEs import *
from hmpinn.utils import get_PDE_object

# torch import
import torch

# hidden_layers = [64, 64, 64, 64, 64, 64]
# poisson_equation = "non_const_BC"
# batch_size = 128
# epochs = 15000
# optimizer = "Adam"
# optimizer_threshold = 7000
# loss_BC_weight = 30
# seed = 0
# boundary_batch_ratio = 0.25
# activation_function = "tanh"


@hydra.main(version_base=None, config_path="config_files", config_name="config_2.yaml")
def main(cfg: DictConfig):
# def main():
    # Generate an instance of the Poisson equation depending on the input
    print("here")
    
    poisson_eq = get_PDE_object(cfg.poisson_equation)

    # Create the model
    model = ModelV2(PDE=poisson_eq,
                     nodes_hidden_layers=cfg.hidden_layers,
                     activation_function=cfg.activation_function)

    # Train the model
    errors, grad_errors, loss, BC_loss = train(model, n_epochs=cfg.epochs, 
                                            batch_size=cfg.batch_size,
                                            optimizer_threshold=cfg.optimizer_threshold,
                                            optimizer=cfg.optimizer,
                                            loss_BC_weight=cfg.loss_BC_weight,
                                            boundary_batch_ratio=cfg.boundary_batch_ratio,
                                            save_BC_loss=True,
                                            seed=cfg.seed)

    # Save the errors and the model
    torch.save(model, "model.pth")

    # Save the errors and losses
    torch.save(errors, "errors.pt")
    torch.save(grad_errors, "grad_errors.pt")
    torch.save(loss, "loss.pt")
    torch.save(BC_loss, "BC_loss.pt")

    

if __name__ == "__main__":
    main()