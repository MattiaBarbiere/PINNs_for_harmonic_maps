#hydra imports
import hydra
from omegaconf import DictConfig

# project imports
from hmpinn.models.model_v1 import ModelV1
from hmpinn.utils.ml_utils import train
from hmpinn.utils import get_PDE_object

# torch import
import torch


@hydra.main(version_base=None, config_path="config_files", config_name="config_1.yaml")
def main(cfg: DictConfig):
    # Generate an instance of the Poisson equation depending on the input
    param_dict = {"PDE": {"name": cfg.poisson_equation, "PDE_kwargs": {}}}

    poi_eq = get_PDE_object(param_dict, backend=torch)

    # Create the model
    model = ModelV1(PDE=poi_eq, 
                     embedding_size_per_dim=cfg.embeddings_per_dim, 
                     nodes_hidden_layers=cfg.hidden_layers)

    # Train the model
    errors, grad_errors, loss = train(model, n_epochs=cfg.epochs, 
                                            batch_size=cfg.batch_size,
                                            optimizer_threshold=cfg.optimizer_threshold,
                                            optimizer=cfg.optimizer,
                                            save_BC_loss=False)

    # Save the errors and the model
    torch.save(model.state_dict(), "model.pt")

    # Save the errors and losses
    torch.save(errors, "errors.pt")
    torch.save(grad_errors, "grad_errors.pt")
    torch.save(loss, "loss.pt")

    

if __name__ == "__main__":
    main()