from .wan import PD_adam_WAN_solver
from .deepritz import DeepRitz_solver
from .pinn import PINN_solver
from .npdhg import PDHG_solver_extrapolation_in_func_space_Bdry_loss

__all__ = [
    "PD_adam_WAN_solver",
    "DeepRitz_solver",
    "PINN_solver",
    "PDHG_solver_extrapolation_in_func_space_Bdry_loss",
]
