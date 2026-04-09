from .hierarchical_solver import HierarchicalSolver
from .global_opt_solver import GlobalOptSolver
from .repair_solver import RepairSolver
from .llm_codegen_solver import LLMCodegenSolver
from .base import BaseSolver, SolverResult

SOLVER_REGISTRY = {
    "hierarchical": HierarchicalSolver,
    "global_opt": GlobalOptSolver,
    "repair": RepairSolver,
    "llm_codegen": LLMCodegenSolver,
}
