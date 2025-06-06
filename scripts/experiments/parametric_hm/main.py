from ml_utils import *
from model import ParametricHmModel


model = ParametricHmModel()

errors, grad_errors, loss, BC_loss = train_parametric_hm(model, n_epochs=15000, 
                                            batch_size=32,
                                            optimizer_threshold=8000,
                                            optimizer="Adam",
                                            loss_BC_weight=20,
                                            save_BC_loss=True)

# Save the errors and the model
torch.save(model.state_dict(), "model.pt")
# Save the errors and losses
torch.save(errors, "errors.pt")
torch.save(grad_errors, "grad_errors.pt")
torch.save(loss, "loss.pt")
if BC_loss is not None:
    torch.save(BC_loss, "BC_loss.pt")