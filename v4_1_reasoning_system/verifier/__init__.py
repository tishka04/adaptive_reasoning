from .base import BaseVerifier, VerificationResult
from .planning_verifier import PlanningVerifier
from .scheduling_verifier import SchedulingVerifier
from .dispatch_verifier import DispatchVerifier
from .code_verifier import CodeVerifier

VERIFIER_REGISTRY = {
    "planning": PlanningVerifier,
    "scheduling": SchedulingVerifier,
    "optimization": DispatchVerifier,
    "coding": CodeVerifier,
}
