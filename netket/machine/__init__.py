from .abstract_machine import AbstractMachine

from .rbm import RbmSpin, RbmSpinReal, RbmSpinSymm, RbmMultiVal, RbmSpinPhase, RbmDimer
from .jastrow import Jastrow, JastrowSymm
from ..utils import jax_available, torch_available


if jax_available:
    from .jax import Jax, JaxRbm, MPSPeriodic, JaxRbmSpinPhase
    from .jax import SumLayer, LogCoshLayer

if torch_available:
    from .torch import Torch, TorchLogCosh, TorchView


from . import density_matrix
from .functions import graph_hex_4 as graph_hex
from .functions2 import new_hex