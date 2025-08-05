from .base import BasePrivacy
from .federated import FederatedLearning
from .differential import DifferentialPrivacy
from .smpc import SMPC

# Auto-register privacy techniques with registry
from core.registry import registry

registry.register_privacy('federated_learning', FederatedLearning)
registry.register_privacy('differential_privacy', DifferentialPrivacy)
registry.register_privacy('secure_multiparty_computation', SMPC)

__all__ = ['BasePrivacy', 'FederatedLearning', 'DifferentialPrivacy', 'SMPC']
