import json
import logging
import os
import re
import sys
from pathlib import Path

# JSON config persisted alongside this file
_CONFIG_PATH = Path(__file__).parent / "miopen_config.json"


# ---------------------------------------------------------------------------
# Path helpers  (mirrors rocm_mgr._expand_venv / _collapse_venv)
# ---------------------------------------------------------------------------

def _expand_vars(val: str) -> str:
    """Expand {VIRTUAL_ENV} and {ROOT} tokens in path strings."""
    try:
        import folder_paths  # available when running inside ComfyUI
        root = folder_paths.base_path
    except Exception:
        root = ""
    return val.replace("{VIRTUAL_ENV}", sys.prefix).replace("{ROOT}", root)


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as _m
        return _m.version(name)
    except Exception:
        return "n/a"


def _hip_version_from_file(db_path: Path) -> str:
    hip_ver_file = db_path / ".hipVersion"
    if not hip_ver_file.exists():
        return ""
    kv = {}
    for line in hip_ver_file.read_text(errors="ignore").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()
    major = kv.get("HIP_VERSION_MAJOR", "")
    minor = kv.get("HIP_VERSION_MINOR", "")
    patch = kv.get("HIP_VERSION_PATCH", "")
    git   = kv.get("HIP_VERSION_GITHASH", "")
    if major:
        return f"{major}.{minor}.{patch} ({git})"
    return ""


def _user_db_summary(path: Path) -> dict:
    out = {}
    for pat in ("*.udb.txt", "*.ufdb.txt"):
        for f in sorted(path.glob(pat)):
            kb = f.stat().st_size // 1024
            try:
                lines = sum(1 for _ in f.open(errors="ignore"))
            except Exception:
                lines = 0
            out[f.name] = f"{kb} KB, {lines} entries"
    return out


def _extract_db_hash(db_path: Path) -> str:
    for f in db_path.glob("*.HIP.*.udb.txt"):
        m = re.search(r'\.HIP\.([^.]+)\.udb\.txt$', f.name)
        if m:
            return m.group(1).replace("_", ".")
    return ""


def _db_file_summary(path: Path, patterns: list) -> dict:
    out = {}
    for pat in patterns:
        for f in sorted(path.glob(pat)):
            kb = f.stat().st_size // 1024
            out[f.name] = f"{kb} KB"
    return out


def _user_cache_summary(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for f in sorted(path.iterdir()):
        if f.is_file():
            kb = f.stat().st_size // 1024
            out[f.name] = f"{kb} KB"
    return out


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


any_type = AlwaysEqualProxy("*")

# Vars that must never be set — dtype-corrupting or config strings, not booleans.
_UNSAFE_VARS = {
    "MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL",
    "MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL",
    "MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_PK_ATOMIC_ADD_FP16",
    "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS",
    "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS",
    "HIP_PATH",
    "HIP_PATH_71",
}


# ---------------------------------------------------------------------------
# CFZ_MIOpen_Settings — GEMM backend, find mode/enforce, rocBLAS/hipBLASLt, logging
# ---------------------------------------------------------------------------

class CFZ_MIOpen_Settings:
    """Configure MIOpen GEMM backend, find mode, rocBLAS/hipBLASLt, TunableOp, and logging.

    load_config_on_run  — apply miopen_config.json; widget values are ignored.
    save_config_on_run  — save current env state to miopen_config.json after applying.
    delete_saved_config — delete miopen_config.json when run.
    Info output         — compact system/path summary; wire to a Show Text node."""

    @classmethod
    def INPUT_TYPES(cls):
        cfg = _read_config()
        def _b(key, fallback):
            return cfg.get(key, "1" if fallback else "0") == "1"
        def _i(key, fallback):
            try:
                return int(cfg.get(key, fallback))
            except (ValueError, TypeError):
                return fallback
        gemm_raw = cfg.get("MIOPEN_GEMM_ENFORCE_BACKEND", "1")
        gemm_choices = ["hipBLASLt", "rocBLAS"] if gemm_raw == "5" else ["rocBLAS", "hipBLASLt"]
        return {
            "required": {
                "load_config_on_run":                  ("BOOLEAN", {"default": False}),
                "save_config_on_run":                  ("BOOLEAN", {"default": False}),
                "delete_saved_config":                 ("BOOLEAN", {"default": False}),
                "gemm_backend":                        (gemm_choices,),
                "pytorch_rocm_use_rocblas":             ("BOOLEAN", {"default": _b("PYTORCH_ROCM_USE_ROCBLAS",               False)}),
                "pytorch_hipblaslt_disable":            ("BOOLEAN", {"default": _b("PYTORCH_HIPBLASLT_DISABLE",              True)}),
                "rocblas_use_hipblaslt":                ("BOOLEAN", {"default": _b("ROCBLAS_USE_HIPBLASLT",                  False)}),
                "miopen_find_mode":                     ("INT",     {"default": _i("MIOPEN_FIND_MODE",    2), "min": 1, "max": 7, "step": 1}),
                "miopen_find_enforce":                  ("INT",     {"default": _i("MIOPEN_FIND_ENFORCE", 1), "min": 1, "max": 5, "step": 1}),
                "miopen_search_cutoff":                 ("BOOLEAN", {"default": _b("MIOPEN_SEARCH_CUTOFF",                   False)}),
                "miopen_deterministic_conv":            ("BOOLEAN", {"default": _b("MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC",  False)}),
                "rocblas_stream_order_alloc":           ("BOOLEAN", {"default": _b("ROCBLAS_STREAM_ORDER_ALLOC",              True)}),
                "rocblas_default_atomics_mode":         ("BOOLEAN", {"default": _b("ROCBLAS_DEFAULT_ATOMICS_MODE",            True)}),
                "pytorch_tunableop_rocblas_enabled":    ("BOOLEAN", {"default": _b("PYTORCH_TUNABLEOP_ROCBLAS_ENABLED",       False)}),
                "pytorch_tunableop_hipblaslt_enabled":  ("BOOLEAN", {"default": _b("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED",     False)}),
                "miopen_log_level":                     ("INT",     {"default": _i("MIOPEN_LOG_LEVEL",    0), "min": 0, "max": 7, "step": 1}),
                "miopen_debug_enable":                  ("BOOLEAN", {"default": _b("MIOPEN_DEBUG_ENABLE",                     False)}),
            },
            "optional": {
                "trigger": (any_type, {}),
            },
        }

    RETURN_TYPES = ("STRING", any_type)
    RETURN_NAMES = ("info",    "output")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "CFZ Utils/MIOpen"

    def _build_info(self) -> str:
        lines = []
        def row(k, v):
            lines.append(f"  {k:<40} {v}")

        lines.append("=== ROCm / System Info ===")
        hip_ver = ""
        try:
            import torch
            hip_ver = getattr(torch.version, "hip", "") or ""
            row("Torch", torch.__version__)
        except Exception:
            pass
        if not hip_ver:
            raw_db = os.environ.get("MIOPEN_SYSTEM_DB_PATH", "")
            if raw_db:
                hip_ver = _hip_version_from_file(Path(_expand_vars(raw_db)))
        row("HIP", hip_ver or "n/a")

        for lbl, env_key in (
            ("System DB", "MIOPEN_SYSTEM_DB_PATH"),
            ("rocBLAS lib", "ROCBLAS_TENSILE_LIBPATH"),
            ("TunableOp cache", "PYTORCH_TUNABLEOP_CACHE_DIR"),
        ):
            raw = os.environ.get(env_key, "")
            if raw:
                p = Path(_expand_vars(raw))
                row(lbl, f"{p}  [{'EXISTS' if p.exists() else 'NOT FOUND'}]")
            else:
                row(lbl, "(not set — run CFZ MIOpen Paths first)")

        user_db = Path.home() / ".miopen" / "db"
        if user_db.exists():
            n = len(list(user_db.glob("*.udb.txt")) + list(user_db.glob("*.ufdb.txt")))
            row("User DB", f"{user_db}  [{n} file(s)]")
        else:
            row("User DB", f"{user_db}  [not present]")

        cache_base = Path.home() / ".miopen" / "cache"
        db_hash = _extract_db_hash(user_db) if user_db.exists() else ""
        cache_path = cache_base / db_hash if db_hash else cache_base
        row("User cache", f"{cache_path}  [{'EXISTS' if cache_path.exists() else 'not present'}]")

        row("Config file", f"{_CONFIG_PATH}  [{'SAVED' if _CONFIG_PATH.exists() else 'not saved'}]")
        return "\n".join(lines)

    def run(self, load_config_on_run, save_config_on_run, delete_saved_config,
            gemm_backend, pytorch_rocm_use_rocblas, pytorch_hipblaslt_disable,
            rocblas_use_hipblaslt, miopen_find_mode, miopen_find_enforce,
            miopen_search_cutoff, miopen_deterministic_conv,
            rocblas_stream_order_alloc, rocblas_default_atomics_mode,
            pytorch_tunableop_rocblas_enabled, pytorch_tunableop_hipblaslt_enabled,
            miopen_log_level, miopen_debug_enable, trigger=None):

        # Delete saved config if requested
        if delete_saved_config:
            if _CONFIG_PATH.exists():
                _CONFIG_PATH.unlink()
                logging.info(f"[CFZ MIOpen Settings] Deleted saved config: {_CONFIG_PATH}")
            else:
                logging.info(f"[CFZ MIOpen Settings] No saved config to delete at {_CONFIG_PATH}")

        # Load from JSON or apply widget values
        using_saved = False
        if load_config_on_run:
            cfg = _read_config()
            if cfg:
                count = _apply_config(cfg)
                logging.info(f"[CFZ MIOpen Settings] Loaded {count} var(s) from {_CONFIG_PATH}")
                using_saved = True
            else:
                logging.info(f"[CFZ MIOpen Settings] No saved config at {_CONFIG_PATH} — applying widget values")

        if not using_saved:
            def b(v):
                return "1" if v else "0"
            gemm_val = "1" if gemm_backend == "rocBLAS" else "5"
            os.environ["MIOPEN_GEMM_ENFORCE_BACKEND"]            = gemm_val
            os.environ["PYTORCH_ROCM_USE_ROCBLAS"]               = b(pytorch_rocm_use_rocblas)
            os.environ["PYTORCH_HIPBLASLT_DISABLE"]              = b(pytorch_hipblaslt_disable)
            os.environ["ROCBLAS_USE_HIPBLASLT"]                  = b(rocblas_use_hipblaslt)
            os.environ["MIOPEN_FIND_MODE"]                       = str(miopen_find_mode)
            os.environ["MIOPEN_FIND_ENFORCE"]                    = str(miopen_find_enforce)
            os.environ["MIOPEN_SEARCH_CUTOFF"]                   = b(miopen_search_cutoff)
            os.environ["MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC"] = b(miopen_deterministic_conv)
            os.environ["ROCBLAS_STREAM_ORDER_ALLOC"]             = b(rocblas_stream_order_alloc)
            os.environ["ROCBLAS_DEFAULT_ATOMICS_MODE"]           = b(rocblas_default_atomics_mode)
            os.environ["PYTORCH_TUNABLEOP_ROCBLAS_ENABLED"]      = b(pytorch_tunableop_rocblas_enabled)
            os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"]    = b(pytorch_tunableop_hipblaslt_enabled)
            os.environ["MIOPEN_LOG_LEVEL"]                       = str(miopen_log_level)
            os.environ["MIOPEN_DEBUG_ENABLE"]                    = b(miopen_debug_enable)
            logging.info(
                f"[CFZ MIOpen Settings] Applied: GEMM={gemm_backend}"
                f" FindMode={miopen_find_mode} FindEnforce={miopen_find_enforce}"
                f" LogLevel={miopen_log_level}"
            )
            # Always persist widget values so startup auto-apply works next run
            cfg = {var: val for var in _MANAGED_VARS if (val := os.environ.get(var)) is not None}
            _write_config(cfg)
            logging.info(f"[CFZ MIOpen Settings] Auto-saved {len(cfg)} var(s) to {_CONFIG_PATH}")

        # Save current env state to JSON if explicitly requested
        elif save_config_on_run:
            cfg = {}
            for var in _MANAGED_VARS:
                val = os.environ.get(var)
                if val is not None:
                    cfg[var] = val
            _write_config(cfg)
            logging.info(f"[CFZ MIOpen Settings] Saved {len(cfg)} var(s) to {_CONFIG_PATH}")

        info = self._build_info()
        print(info)
        return (info, trigger)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


# ---------------------------------------------------------------------------
# Solver node base — shared logic for all per-group solver nodes
# ---------------------------------------------------------------------------

def _make_solver_node(class_name: str, display_title: str, solver_map: tuple):
    """Factory that returns a solver node class for one solver group."""

    def _input_types(cls):
        cfg = _read_config()
        required = {}
        for param, env_var, default in solver_map:
            val = (cfg[env_var] == "1") if env_var in cfg else default
            required[param] = ("BOOLEAN", {"default": val})
        return {"required": required, "optional": {"trigger": (any_type, {})}}

    def _run(self, **kwargs):
        trigger = kwargs.pop("trigger", None)
        on, off = 0, 0
        for param, env_var, _default in solver_map:
            val = kwargs.get(param, _default)
            os.environ[env_var] = "1" if val else "0"
            if val: on += 1
            else:   off += 1
        logging.info(f"[{class_name}] {on} enabled, {off} disabled")
        return (trigger,)

    return type(class_name, (), {
        "_SOLVER_MAP":    solver_map,
        "INPUT_TYPES":    classmethod(_input_types),
        "RETURN_TYPES":   (any_type,),
        "RETURN_NAMES":   ("output",),
        "OUTPUT_NODE":    True,
        "FUNCTION":       "run",
        "CATEGORY":       "CFZ Utils/MIOpen/Solvers",
        "run":            _run,
        "IS_CHANGED":     classmethod(lambda cls, **kw: float("NaN")),
    })


# All solver tuples in one place — referenced by group nodes and _MANAGED_VARS
_SOLVERS_ALGO = (
    ("conv_fft",           "MIOPEN_DEBUG_CONV_FFT",           True),
    ("conv_direct",        "MIOPEN_DEBUG_CONV_DIRECT",        True),
    ("conv_gemm",          "MIOPEN_DEBUG_CONV_GEMM",          True),
    ("conv_winograd",      "MIOPEN_DEBUG_CONV_WINOGRAD",      True),
    ("conv_implicit_gemm", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM", True),
)
_SOLVERS_FALLBACK = (
    ("conv_immed_fallback",  "MIOPEN_DEBUG_CONV_IMMED_FALLBACK",             True),
    ("ai_immed_fallback",    "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK",   True),
    ("force_immed_fallback", "MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK",       False),
)
_SOLVERS_BUILD = (
    ("gcn_asm_kernels",     "MIOPEN_DEBUG_GCN_ASM_KERNELS",      True),
    ("hip_kernels",         "MIOPEN_DEBUG_HIP_KERNELS",          True),
    ("opencl_convolutions", "MIOPEN_DEBUG_OPENCL_CONVOLUTIONS",  True),
    ("opencl_wave64_nowgp", "MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP",  True),
    ("attn_softmax",        "MIOPEN_DEBUG_ATTN_SOFTMAX",         True),
)
_SOLVERS_DIRECT_ASM = (
    ("direct_asm_3x3u",        "MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U",                  True),
    ("direct_asm_1x1u",        "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U",                  True),
    ("direct_asm_1x1uv2",      "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2",                True),
    ("direct_asm_1x1u_search", "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED", True),
    ("direct_asm_1x1u_ai",     "MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR",          True),
)
_SOLVERS_DIRECT_OCL = (
    ("direct_ocl_fwd",    "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD",    True),
    ("direct_ocl_fwd1x1", "MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1", True),
)
_SOLVERS_WINOGRAD = (
    ("winograd_3x3",        "MIOPEN_DEBUG_AMD_WINOGRAD_3X3",           True),
    ("winograd_rxs",        "MIOPEN_DEBUG_AMD_WINOGRAD_RXS",           True),
    ("winograd_rxs_fwdbwd", "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD",   True),
    ("winograd_rxs_f3x2",   "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2",      True),
    ("winograd_rxs_f2x3",   "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3",      True),
    ("winograd_rxs_f2x3_g1","MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1",   True),
    ("winograd_fused",      "MIOPEN_DEBUG_AMD_FUSED_WINOGRAD",         True),
    ("winograd_fury_f2x3",  "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3", False),
    ("winograd_fury_f3x2",  "MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2", False),
    ("winograd_rage_f2x3",  "MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3", False),
    ("mpass_f3x2",          "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2",    True),
    ("mpass_f3x3",          "MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3",    True),
)
_SOLVERS_IGEMM = (
    ("igemm_asm_v4r1",     "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1",     True),
    ("igemm_asm_v4r1_1x1", "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1", True),
    ("igemm_hip_v4r1",     "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1",     True),
    ("igemm_hip_v4r4",     "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4",     True),
)
_SOLVERS_CK = (
    ("group_conv_xdlops",    "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS",         False),
    ("group_conv_xdlops_ai", "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR", False),
    ("ck_default_kernels",   "MIOPEN_DEBUG_CK_DEFAULT_KERNELS",                               False),
)

# All solver tuples flat — used by _MANAGED_VARS
_ALL_SOLVER_TUPLES = (
    _SOLVERS_ALGO + _SOLVERS_FALLBACK + _SOLVERS_BUILD +
    _SOLVERS_DIRECT_ASM + _SOLVERS_DIRECT_OCL +
    _SOLVERS_WINOGRAD + _SOLVERS_IGEMM + _SOLVERS_CK
)

CFZ_MIOpen_Solvers          = _make_solver_node("CFZ_MIOpen_Solvers",          "Algorithm / Solver Groups",       _SOLVERS_ALGO)
CFZ_MIOpen_SolversFallback  = _make_solver_node("CFZ_MIOpen_SolversFallback",  "Immediate Fallback Mode",         _SOLVERS_FALLBACK)
CFZ_MIOpen_SolversBuild     = _make_solver_node("CFZ_MIOpen_SolversBuild",     "Build Method Toggles",            _SOLVERS_BUILD)
CFZ_MIOpen_SolversDirectASM = _make_solver_node("CFZ_MIOpen_SolversDirectASM", "Direct ASM Solver Toggles",       _SOLVERS_DIRECT_ASM)
CFZ_MIOpen_SolversDirectOCL = _make_solver_node("CFZ_MIOpen_SolversDirectOCL", "Direct OpenCL Solver Toggles",    _SOLVERS_DIRECT_OCL)
CFZ_MIOpen_SolversWinograd  = _make_solver_node("CFZ_MIOpen_SolversWinograd",  "Winograd + Multi-pass Toggles",   _SOLVERS_WINOGRAD)
CFZ_MIOpen_SolversIGEMM     = _make_solver_node("CFZ_MIOpen_SolversIGEMM",     "Implicit GEMM Toggles",           _SOLVERS_IGEMM)
CFZ_MIOpen_SolversCK        = _make_solver_node("CFZ_MIOpen_SolversCK",        "Group Conv / CK (RDNA3/4+ only)", _SOLVERS_CK)


# ---------------------------------------------------------------------------
# CFZ_MIOpen_Paths — set path / workspace environment variables
# ---------------------------------------------------------------------------

_VENV_DEFAULT_DB      = r"{VIRTUAL_ENV}\Lib\site-packages\_rocm_sdk_devel\bin"
_VENV_DEFAULT_RBLAS   = r"{VIRTUAL_ENV}\Lib\site-packages\_rocm_sdk_devel\bin\rocblas\library"
_ROOT_DEFAULT_TUNABLE = r"{ROOT}\models\tunable"


class CFZ_MIOpen_Paths:
    """Set MIOpen / rocBLAS path and workspace-size environment variables.

    Use {VIRTUAL_ENV} for the active venv prefix and {ROOT} for the ComfyUI root.
    Leave a field blank to leave that variable unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        cfg = _read_config()
        def _s(key, fallback):
            return cfg.get(key, fallback)
        return {
            "required": {
                "miopen_system_db_path":            ("STRING", {"default": _s("MIOPEN_SYSTEM_DB_PATH",            _expand_vars(_VENV_DEFAULT_DB)),      "multiline": False}),
                "rocblas_tensile_libpath":           ("STRING", {"default": _s("ROCBLAS_TENSILE_LIBPATH",          _expand_vars(_VENV_DEFAULT_RBLAS)),   "multiline": False}),
                "pytorch_tunableop_cache_dir":       ("STRING", {"default": _s("PYTORCH_TUNABLEOP_CACHE_DIR",      _expand_vars(_ROOT_DEFAULT_TUNABLE)), "multiline": False}),
                "miopen_convolution_max_workspace":  ("STRING", {"default": _s("MIOPEN_CONVOLUTION_MAX_WORKSPACE", "1073741824"),                        "multiline": False}),
                "rocblas_device_memory_size":        ("STRING", {"default": _s("ROCBLAS_DEVICE_MEMORY_SIZE",       ""),                                  "multiline": False}),
            },
            "optional": {
                "trigger": (any_type, {}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "CFZ Utils/MIOpen"

    def run(self, miopen_system_db_path, rocblas_tensile_libpath,
            pytorch_tunableop_cache_dir, miopen_convolution_max_workspace,
            rocblas_device_memory_size, trigger=None):

        assignments = {
            "MIOPEN_SYSTEM_DB_PATH":            miopen_system_db_path,
            "ROCBLAS_TENSILE_LIBPATH":          rocblas_tensile_libpath,
            "PYTORCH_TUNABLEOP_CACHE_DIR":      pytorch_tunableop_cache_dir,
            "MIOPEN_CONVOLUTION_MAX_WORKSPACE": miopen_convolution_max_workspace,
            "ROCBLAS_DEVICE_MEMORY_SIZE":       rocblas_device_memory_size,
        }

        logging.info("[CFZ MIOpen Paths]")
        for var, raw in assignments.items():
            if raw.strip() == "":
                logging.info(f"  {var}: (skipped — empty)")
                continue
            expanded = _expand_vars(raw.strip())
            os.environ[var] = expanded
            exists = Path(expanded).exists() if expanded else False
            status = "OK" if exists else "NOT FOUND"
            logging.info(f"  {var} = {expanded}  [{status}]")

        return (trigger,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


# ---------------------------------------------------------------------------
# CFZ_MIOpen_DBInfo — inspect system DB, user DB and user cache on-disk
# ---------------------------------------------------------------------------

class CFZ_MIOpen_DBInfo:
    """Read and report MIOpen database and cache locations.

    Reads MIOPEN_SYSTEM_DB_PATH from the environment (or the override input),
    then scans the system DB, user DB (~/.miopen/db) and user cache
    (~/.miopen/cache) and prints a human-readable summary to the console.
    Returns the summary as a STRING for optional display nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "system_db_path_override": ("STRING", {"default": "", "multiline": False}),
                "trigger": (any_type, {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "CFZ Utils/MIOpen"

    def run(self, system_db_path_override="", trigger=None):
        lines = []

        def section(title):
            lines.append(f"\n{'='*60}")
            lines.append(f"  {title}")
            lines.append(f"{'='*60}")

        def row(key, value):
            lines.append(f"  {key:<44} {value}")

        # --- Resolve system DB path ---
        raw_db = system_db_path_override.strip() or os.environ.get("MIOPEN_SYSTEM_DB_PATH", "")
        db_path = Path(_expand_vars(raw_db)) if raw_db else Path("")

        # --- ROCm / HIP ---
        section("ROCm / HIP")
        hip_ver = _hip_version_from_file(db_path) if db_path.name else ""
        if not hip_ver:
            try:
                import torch
                hip_ver = getattr(torch.version, "hip", "") or ""
            except Exception:
                pass
        row("HIP version", hip_ver or "n/a")
        for pkg in ("rocm", "rocm-sdk-core", "rocm-sdk-devel"):
            v = _pkg_version(pkg)
            if v != "n/a":
                row(pkg, v)
        libs_pkg = _pkg_version("rocm-sdk-libraries-gfx103x-dgpu")
        if libs_pkg != "n/a":
            row("rocm-sdk-libraries (gfx103x)", libs_pkg)
        try:
            import torch
            row("torch version", torch.__version__)
            row("torch.version.hip", getattr(torch.version, "hip", None) or "n/a")
        except Exception:
            pass

        # --- System DB ---
        section("System DB  (MIOPEN_SYSTEM_DB_PATH)")
        row("path", str(db_path) if db_path.name else "(not set)")
        if db_path.name and db_path.exists():
            for fname, sz in _db_file_summary(db_path, ["*.db.txt"]).items():
                row("  solver_db", f"{fname}  {sz}")
            for fname, sz in _db_file_summary(db_path, ["*.HIP.fdb.txt", "*.fdb.txt"]).items():
                row("  find_db", f"{fname}  {sz}")
            for fname, sz in _db_file_summary(db_path, ["*.kdb"]).items():
                row("  kernel_db", f"{fname}  {sz}")
        elif db_path.name:
            row("status", "NOT FOUND")
        else:
            row("status", "env var not set — run CFZ_MIOpen_Paths first")

        # --- rocBLAS Tensile lib path ---
        section("rocBLAS Tensile Library  (ROCBLAS_TENSILE_LIBPATH)")
        raw_rbl = os.environ.get("ROCBLAS_TENSILE_LIBPATH", "")
        rbl_path = Path(_expand_vars(raw_rbl)) if raw_rbl else Path("")
        row("path", str(rbl_path) if rbl_path.name else "(not set)")
        if rbl_path.name:
            row("exists", "YES" if rbl_path.exists() else "NOT FOUND")

        # --- TunableOp cache dir ---
        section("TunableOp Cache  (PYTORCH_TUNABLEOP_CACHE_DIR)")
        raw_tunable = os.environ.get("PYTORCH_TUNABLEOP_CACHE_DIR", "")
        tunable_path = Path(_expand_vars(raw_tunable)) if raw_tunable else Path("")
        row("path", str(tunable_path) if tunable_path.name else "(not set)")
        if tunable_path.name:
            if tunable_path.exists():
                files = [f for f in sorted(tunable_path.iterdir()) if f.is_file()]
                row("exists", f"YES  ({len(files)} file(s))")
                for f in files[:10]:
                    kb = f.stat().st_size // 1024
                    row("  " + f.name, f"{kb} KB")
                if len(files) > 10:
                    row("  ...", f"({len(files) - 10} more)")
            else:
                row("exists", "NOT FOUND  (will be created on first tuning run)")

        # --- User DB ---
        section("User DB  (~/.miopen/db)")
        user_db_path = Path.home() / ".miopen" / "db"
        row("path", str(user_db_path))
        row("exists", "YES" if user_db_path.exists() else "NO")
        if user_db_path.exists():
            for fname, info_str in _user_db_summary(user_db_path).items():
                row("  " + fname, info_str)

        # --- User cache ---
        section("User Cache  (~/.miopen/cache)")
        cache_base = Path.home() / ".miopen" / "cache"
        db_hash = _extract_db_hash(user_db_path) if user_db_path.exists() else ""
        cache_path = cache_base / db_hash if db_hash else cache_base
        row("path", str(cache_path))
        row("hash", db_hash or "(not derived — user DB empty or missing)")
        row("exists", "YES" if cache_path.exists() else "NO")
        if cache_path.exists():
            for fname, sz in _user_cache_summary(cache_path).items():
                row("  " + fname, sz)

        output = "\n".join(lines)
        logging.info("[CFZ MIOpen DBInfo]" + output)
        return (output,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


# ---------------------------------------------------------------------------
# CFZ_CuDNN — enable/disable cuDNN (MIOpen) and related flags at runtime
# ---------------------------------------------------------------------------

class CFZ_CuDNN:
    """Enable or disable torch.backends.cudnn at runtime.

    On ROCm, 'cuDNN' maps to MIOpen — disabling it forces PyTorch to use
    its own fallback math kernels instead of MIOpen convolution solvers.
    Changes take effect immediately for the current ComfyUI session."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import torch
            default_enabled       = torch.backends.cudnn.enabled
            default_deterministic = torch.backends.cudnn.deterministic
            default_allow_tf32    = torch.backends.cudnn.allow_tf32
        except Exception:
            default_enabled, default_deterministic, default_allow_tf32 = True, False, True
        return {
            "required": {
                "cudnn_enabled":       ("BOOLEAN", {"default": default_enabled}),
                "cudnn_deterministic": ("BOOLEAN", {"default": default_deterministic}),
                "cudnn_allow_tf32":    ("BOOLEAN", {"default": default_allow_tf32}),
            },
            "optional": {
                "trigger": (any_type, {}),
            },
        }

    RETURN_TYPES = ("STRING", any_type)
    RETURN_NAMES = ("info",    "output")
    OUTPUT_NODE  = True
    FUNCTION     = "run"
    CATEGORY     = "CFZ Utils/MIOpen"

    def run(self, cudnn_enabled, cudnn_deterministic, cudnn_allow_tf32, trigger=None):
        try:
            import torch
            torch.backends.cudnn.enabled       = cudnn_enabled
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.allow_tf32    = cudnn_allow_tf32
            info = (
                f"torch.backends.cudnn:\n"
                f"  enabled       = {torch.backends.cudnn.enabled}\n"
                f"  deterministic = {torch.backends.cudnn.deterministic}\n"
                f"  allow_tf32    = {torch.backends.cudnn.allow_tf32}"
            )
        except Exception as e:
            info = f"[CFZ CuDNN] Error: {e}"
        logging.info(f"[CFZ CuDNN] {info}")
        return (info, trigger)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


# ---------------------------------------------------------------------------
# CFZ_CuDNN_Benchmark — control cudnn.benchmark and benchmark_limit
# ---------------------------------------------------------------------------

class CFZ_CuDNN_Benchmark:
    """Control torch.backends.cudnn.benchmark at runtime.

    When enabled, PyTorch (MIOpen on ROCm) benchmarks multiple convolution
    algorithms on the first run and caches the fastest one per input shape.
    benchmark_limit caps how many algorithm candidates are evaluated
    (0 = unlimited)."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import torch
            default_benchmark       = torch.backends.cudnn.benchmark
            default_benchmark_limit = torch.backends.cudnn.benchmark_limit
        except Exception:
            default_benchmark, default_benchmark_limit = False, 10
        return {
            "required": {
                "benchmark":       ("BOOLEAN", {"default": default_benchmark}),
                "benchmark_limit": ("INT",     {"default": default_benchmark_limit, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "trigger": (any_type, {}),
            },
        }

    RETURN_TYPES = ("STRING", any_type)
    RETURN_NAMES = ("info",    "output")
    OUTPUT_NODE  = True
    FUNCTION     = "run"
    CATEGORY     = "CFZ Utils/MIOpen"

    def run(self, benchmark, benchmark_limit, trigger=None):
        try:
            import torch
            torch.backends.cudnn.benchmark       = benchmark
            torch.backends.cudnn.benchmark_limit = benchmark_limit
            info = (
                f"torch.backends.cudnn:\n"
                f"  benchmark       = {torch.backends.cudnn.benchmark}\n"
                f"  benchmark_limit = {torch.backends.cudnn.benchmark_limit}"
            )
        except Exception as e:
            info = f"[CFZ CuDNN Benchmark] Error: {e}"
        logging.info(f"[CFZ CuDNN Benchmark] {info}")
        return (info, trigger)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


# ---------------------------------------------------------------------------
# Config I/O helpers
# ---------------------------------------------------------------------------

# Complete list of every env var our nodes manage — used for save/load scoping.
_MANAGED_VARS: list = [
    # CFZ_MIOpen_Paths
    "MIOPEN_SYSTEM_DB_PATH",
    "ROCBLAS_TENSILE_LIBPATH",
    "PYTORCH_TUNABLEOP_CACHE_DIR",
    "MIOPEN_CONVOLUTION_MAX_WORKSPACE",
    "ROCBLAS_DEVICE_MEMORY_SIZE",
    # CFZ_MIOpen_Settings
    "MIOPEN_GEMM_ENFORCE_BACKEND",
    "PYTORCH_ROCM_USE_ROCBLAS",
    "PYTORCH_HIPBLASLT_DISABLE",
    "ROCBLAS_USE_HIPBLASLT",
    "MIOPEN_FIND_MODE",
    "MIOPEN_FIND_ENFORCE",
    "MIOPEN_SEARCH_CUTOFF",
    "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC",
    "ROCBLAS_STREAM_ORDER_ALLOC",
    "ROCBLAS_DEFAULT_ATOMICS_MODE",
    "PYTORCH_TUNABLEOP_ROCBLAS_ENABLED",
    "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED",
    "MIOPEN_LOG_LEVEL",
    "MIOPEN_DEBUG_ENABLE",
] + [env_var for _, env_var, _ in _ALL_SOLVER_TUPLES]


def _read_config() -> dict:
    """Load saved config from JSON.  Returns {} if file absent or unreadable."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning(f"[CFZ MIOpen] Warning: could not read config {_CONFIG_PATH}: {e}")
        return {}


def _write_config(cfg: dict) -> None:
    """Persist config dict to JSON."""
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def _apply_config(cfg: dict) -> int:
    """Apply a config dict to os.environ.  Returns number of vars set."""
    # Always clear unsafe vars first
    for var in _UNSAFE_VARS:
        os.environ.pop(var, None)
    count = 0
    for var, val in cfg.items():
        if var not in _MANAGED_VARS:
            continue
        if var in _UNSAFE_VARS:
            continue
        expanded = _expand_vars(str(val))
        if expanded == "":
            os.environ.pop(var, None)
        else:
            os.environ[var] = expanded
            count += 1
    return count


# ---------------------------------------------------------------------------
# Auto-apply saved config on module import (ComfyUI startup)
# ---------------------------------------------------------------------------

_startup_cfg = _read_config()
if _startup_cfg:
    _startup_count = _apply_config(_startup_cfg)
    logging.info(f"[CFZ MIOpen] Startup: applied {_startup_count} saved var(s) from {_CONFIG_PATH}")
else:
    logging.info(f"[CFZ MIOpen] Startup: no saved config at {_CONFIG_PATH} — skipped")
