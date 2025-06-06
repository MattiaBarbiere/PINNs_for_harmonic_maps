import torch
import torch.nn as nn
from hmpinn.differential_operators import *
###################################### Test 1 ######################################

#The function we would like to take the Laplacian of is g(x_1,x_2) = (x_1)^2 * (x_2)^2
def g(x):
    return x[:, 0]**2 * x[:, 1]**2

def grad_g(x):
    return torch.stack([2*x[:, 0]*x[:, 1]**2, 2*x[:, 0]**2*x[:, 1]], dim=1)

#The analytical Laplacian of g is 2*(x_2)^2 + 2*(x_1)^2
def laplacian_g(x):
    return 2*x[:, 1]**2 + 2*x[:, 0]**2

#The diffusion matrix function that returns a torch tensor of size (2,2)
def diffusion_matrix(x, model=None):
    batch_size = x.shape[0]
    diffusion = torch.zeros(batch_size, 2, 2)
    diffusion[:, 0, 0] = x[:, 0]**2
    diffusion[:, 0, 1] = x[:, 0]
    diffusion[:, 1, 0] = x[:, 1]**2
    diffusion[:, 1, 1] = x[:, 1]
    return diffusion.requires_grad_()

def laplacian_g_with_diffusion(x):
    return 6 * x[:,0]**2 * x[:,1]**2 + 10 * x[:,1] * x[:,0]**2 + 8 * x[:,0] * x[:,1]**3


###################################### Test 2 ######################################
#The function we would like to take the Laplacian of is h(x_1,x_2) = 0.5 * ln(x_1^2 + x_2^2)
def h(x):
    return 0.5 * torch.log(x[:, 0]**2 + x[:, 1]**2)

#The analytical Laplacian of h is 0
def laplacian_h(x):
    return torch.zeros_like(x[:, 0])

########################################## Test 3 ######################################
class Test2(nn.Module):
    def __init__(self):
        super(Test2, self).__init__()

    def forward(self, x):
        return torch.stack((x[:, 0]**2 * x[:, 1]**5, x[:, 1]**7 * x[:, 0]**4), dim=1)

def true_jacobian(x):
    return torch.stack([
        torch.stack([2 * x[:, 0] * x[:, 1]**5, 5 * x[:, 0]**2 * x[:, 1]**4], dim=1),
        torch.stack([4 * x[:, 1]**7 * x[:, 0] ** 3, 7 * x[:, 1]**6 * x[:, 0]**4], dim=1)
    ], dim=1)

###################################### Assertions ######################################

if __name__ == "__main__":
    x = torch.rand(100, 2, requires_grad=True)
    assert torch.allclose(Laplacian()(g, x), laplacian_g(x)), "The Laplacian function is incorrect for g(x_1, x_2) = (x_1)^2 * (x_2)^2"
    assert torch.allclose(Laplacian()(h, x), laplacian_h(x), atol=1e-4), "The Laplacian function is incorrect for h(x_1, x_2) = 0.5 * ln(x_1^2 + x_2^2)"
    assert torch.allclose(Laplacian()(g, x, diffusion_matrix), laplacian_g_with_diffusion(x)), "The Laplacian with diffusion function is incorrect for g(x_1, x_2) = (x_1)^2 * (x_2)^2"   
    
    y = g(x)
    assert torch.allclose(Laplacian()(y, x, diffusion_matrix), laplacian_g_with_diffusion(x)), "The Laplacian with diffusion function is incorrect for g(x_1, x_2) = (x_1)^2 * (x_2)^2"
    
    # By definition, the lapacian should be equal to the divergence of the gradient    
    assert torch.allclose(Laplacian()(g, x), Divergence()(Gradient()(g, x), x)), "The Laplacian function is not equal to the divergence of the gradient"
    assert torch.allclose(Laplacian()(h, x), Divergence()(Gradient()(h, x), x), atol=1e-4), "The Laplacian function is not equal to the divergence of the gradient"
    
    assert torch.allclose(Jacobian()(Test2(), x), true_jacobian(x)), "The Jacobian function is incorrect for Test2()"
    
    print("The differential operators function with diffusion works!!!")