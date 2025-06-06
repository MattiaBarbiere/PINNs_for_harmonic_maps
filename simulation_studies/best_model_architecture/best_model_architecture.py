#hydra imports
import hydra
from omegaconf import DictConfig

# project imports
from hmpinn.models.model_v0 import ModelV0
from hmpinn.ML_utils import train
from hmpinn.utils import get_PDE_object

# poisson problem import
from hmpinn.core.Poisson_eq_examples import *

# torch import
import torch


@hydra.main(version_base=None, config_path="config_files", config_name="config_1.yaml")
def main(cfg: DictConfig):
    # Generate an instance of the Poisson equation depending on the input
    poisson_eq = get_PDE_object(cfg.poisson_equation)

    # Create the model
    model = ModelV0(PDE=poisson_eq, 
                     embedding_size_per_dim=cfg.embeddings_per_dim, 
                     has_embedding_layer=cfg.has_embedding_layer,
                     nodes_hidden_layers=cfg.hidden_layers)

    # Train the model
    errors, grad_errors, loss, BC_loss = train(model, n_epochs=cfg.epochs, 
                                            batch_size=cfg.batch_size,
                                            optimizer_threshold=cfg.optimizer_threshold,
                                            optimizer=cfg.optimizer,
                                            loss_BC_weight=cfg.loss_BC_weight,
                                            save_BC_loss=True,
                                            seed=cfg.seed)

    # Save the errors and the model
    torch.save(model.state_dict(), "model.pt")

    # Save the errors and losses
    torch.save(errors, "errors.pt")
    torch.save(grad_errors, "grad_errors.pt")
    torch.save(loss, "loss.pt")
    torch.save(BC_loss, "BC_loss.pt")

    

if __name__ == "__main__":
    main()