from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Configuration:
    code_dir: str
    solver_name: str
    unroll: bool
    prefix: str
    gradient: bool
    gradient_two_stage: bool
    explicit: bool
    cmake_args: list = field(default_factory=list)


@dataclass
class AffineMap:
    mapping_rows: list = field(default_factory=list)
    mapping: list = field(default_factory=list)
    sign: int = 1
    indices: list = field(default_factory=list)
    indptr: list = field(default_factory=list)
    shape = ()


@dataclass
class ParameterCanon:
    """Represents first affine form"""

    p: dict = field(default_factory=dict)
    p_csc: dict[str, sp.csc_matrix] = field(default_factory=dict)
    p_id_to_mapping: dict[str, sp.csr_matrix] = field(default_factory=dict)  # Represents A slice to canonical parameter
    p_id_to_changes: dict[str, bool] = field(default_factory=dict)
    p_id_to_size: dict[str, int] = field(default_factory=dict)
    nonzero_d: bool = True
    is_maximization: bool = False
    user_p_name_to_canon_outdated: dict[str, list[str]] = field(default_factory=dict)
    quad_obj: bool = True
    th_mask: np.ndarray = None
    n_param_reduced: int = 0
    n_dual_reduced: int = 0


@dataclass
class ParameterInfo:
    """All info about a user defined parameter and how to convert from
    the user-defined parameter to the canonicalized vector that is
    passed to A.
    """

    col_to_name_usp: dict[int, str]  # usp: user-defined sparsity
    flat_usp: np.ndarray
    id_to_col: dict[int, int]  # Maps parameter id to column of the start of the parameter
    ids: list[int]
    name_to_shape: dict[str, tuple]
    name_to_size_usp: dict[str, int]
    name_to_sparsity: dict[str, np.ndarray]
    name_to_sparsity_type: dict[str, str]
    names: list[str]
    num: int
    sparsity_mask: np.ndarray
    writable: dict[str, np.ndarray]
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class VariableInfo:
    name_to_offset: dict[str, int]
    name_to_indices: dict[str, np.ndarray]
    name_to_size: dict[str, int]
    sizes: list[int]
    name_to_shape: dict[str, tuple]
    name_to_init: dict[str, np.ndarray]


@dataclass
class PrimalVariableInfo(VariableInfo):
    """Info for primal variable retrival from a canonical solution"""

    name_to_sym: dict[str, bool]
    sym: list[bool]
    reduced: bool = False


@dataclass
class DualVariableInfo(VariableInfo):
    """Info for dual variable retrival from a canonical solution"""

    name_to_vec: dict[str, str]


@dataclass
class ConstraintInfo:
    n_data_constr: int
    n_data_constr_mat: int
    mapping_rows_eq: np.ndarray
    mapping_rows_ineq: np.ndarray


@dataclass
class WorkspacePointerInfo:
    objective_value: str
    iterations: str
    status: str
    primal_residual: str
    dual_residual: str
    primal_solution: str
    dual_solution: str
    settings: str = None


@dataclass
class UpdatePendingLogic:
    parameters_outdated: list[str]
    operator: str = None
    functions_if_false: list[str] = None
    extra_condition: str = None
    extra_condition_operator: str = None


@dataclass
class ParameterUpdateLogic:
    update_pending_logic: UpdatePendingLogic
    function_call: str


@dataclass
class Canon:
    """All info for the ASA representation
    """

    prim_variable_info: PrimalVariableInfo
    dual_variable_info: DualVariableInfo
    parameter_info: ParameterInfo
    parameter_canon: ParameterCanon


@dataclass
class Setting:
    type: str
    default: str
    enabled: bool = True
    name_cvxpy: str = None
