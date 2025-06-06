# Non divergence form for a PDEs
from hmpinn.PDEs.non_div_form_PDEs.constant_source import ConstantSourceNonDF
from hmpinn.PDEs.non_div_form_PDEs.eigenfunction_source import EigenfunctionSourceNonDF
from hmpinn.PDEs.non_div_form_PDEs.inhomo_bc import InhomoBCNonDF
from hmpinn.PDEs.non_div_form_PDEs.guassian_bump import GuassianBumpNonDF
from hmpinn.PDEs.non_div_form_PDEs.non_differentiable_diffusion import NonDifferentiableDiffusion
from hmpinn.PDEs.non_div_form_PDEs.convection_dominated import ConvectionDominatedNonDF
from hmpinn.PDEs.non_div_form_PDEs.non_sym_hessian import NonSymHessian

# Divergence form for a PDEs
from hmpinn.PDEs.div_form_PDEs.constant_source import ConstantSourceDF
from hmpinn.PDEs.div_form_PDEs.eigenfunction_source import EigenfunctionSourceDF
from hmpinn.PDEs.div_form_PDEs.inhomo_bc import InhomoBCDF
from hmpinn.PDEs.div_form_PDEs.non_constant_bc import NonConstantBC
from hmpinn.PDEs.div_form_PDEs.non_sym_diffusion import NonSymDiffusion
from hmpinn.PDEs.div_form_PDEs.sym_diffusion import SymDiffusion
from hmpinn.PDEs.div_form_PDEs.convection_dominated import ConvectionDominatedDF
from hmpinn.PDEs.div_form_PDEs.piecewise_diffusion import PiecewiseDiffusion

# Import harmonic maps
from hmpinn.PDEs.harmonic_maps.quarter_annulus import QuarterAnnulusHM
from hmpinn.PDEs.harmonic_maps.L_bend import LBendHM
from hmpinn.PDEs.harmonic_maps.sin_boundaries import SinBoundariesHM
from hmpinn.PDEs.harmonic_maps.poly_boundaries import PolynomialBoundariesHM

from hmpinn.PDEs.PDE_factory import construct_PDE_class

# A dict that gives every PDE a shortened name
PDE_NAME_TO_CLASS = {
        "diff": NonSymDiffusion,
        "sym_diff": SymDiffusion,
        "eigenfunc": EigenfunctionSourceDF,
        "poly": InhomoBCDF,
        "const_source": ConstantSourceDF,
        "non_const_BC": NonConstantBC,
        "piecewise_diff": PiecewiseDiffusion,
        "eigenfunc_NonDF": EigenfunctionSourceNonDF,
        "const_source_NonDF": ConstantSourceNonDF,
        "poly_NonDF": InhomoBCNonDF,
        "gaussian_bump_NonDF": GuassianBumpNonDF,
        "non_differentiable_diff": NonDifferentiableDiffusion,
        "convection_dominated_NonDF": ConvectionDominatedNonDF,
        "convection_dominated": ConvectionDominatedDF,
        "non_sym_hess": NonSymHessian,
        "quarter_annulus_hm": QuarterAnnulusHM,
        "L_bend_hm": LBendHM,
        "sin_boundaries_hm": SinBoundariesHM,
        "poly_boundaries_hm": PolynomialBoundariesHM,
    }
