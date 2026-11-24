"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Dict

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver
from .det_solver_v1_sparse_eval import DetSolverV1SparseEval

TASKS: Dict[str, BaseSolver] = {
    "classification": ClasSolver,
    "detection": DetSolver,
    "detection_sparse_eval": DetSolverV1SparseEval,
}
