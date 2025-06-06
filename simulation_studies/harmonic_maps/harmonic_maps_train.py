#hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

# project imports
from hmpinn.utils.ml_utils import train
from hmpinn.utils import get_PDE_object, get_model_class

# poisson problem import
from hmpinn.PDEs import *

# torch import
import torch


@hydra.main(version_base=None, config_path="config_files", config_name="sin_boundaries.yaml")
def main(cfg: DictConfig):
    # Make the config into a normal dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Generate an instance of the Poisson equation depending on the input
    poisson_eq = get_PDE_object(cfg_dict, backend=torch)
    model_class = get_model_class(cfg.model.type)

    # Create the model
    model = model_class(PDE=poisson_eq, **cfg.model.model_kwargs)

    # Train the model
    errors, grad_errors, loss, BC_loss = train(model, **cfg.train)

    # Save the errors and the model
    torch.save(model.state_dict(), "model.pt")

    # Save the errors and losses
    torch.save(errors, "errors.pt")
    torch.save(grad_errors, "grad_errors.pt")
    torch.save(loss, "loss.pt")
    torch.save(BC_loss, "BC_loss.pt")



if __name__ == "__main__":
    main()