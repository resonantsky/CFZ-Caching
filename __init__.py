import warnings
warnings.filterwarnings("ignore", message="Should have tb<=t1", module="torchsde")

from .cfz_caching_condition import save_conditioning, load_conditioning, CFZ_PrintMarker
from .cfz_miopen import (
    CFZ_MIOpen_Settings, CFZ_MIOpen_Paths, CFZ_MIOpen_DBInfo,
    CFZ_MIOpen_Solvers, CFZ_MIOpen_SolversFallback, CFZ_MIOpen_SolversBuild,
    CFZ_MIOpen_SolversDirectASM, CFZ_MIOpen_SolversDirectOCL,
    CFZ_MIOpen_SolversWinograd, CFZ_MIOpen_SolversIGEMM, CFZ_MIOpen_SolversCK,
    CFZ_CuDNN, CFZ_CuDNN_Benchmark,
)

NODE_CLASS_MAPPINGS = {
    "CFZ_save_conditioning":      save_conditioning,
    "CFZ_load_conditioning":      load_conditioning,
    "CFZ_PrintMarker":            CFZ_PrintMarker,
    "CFZ_MIOpen_Settings":        CFZ_MIOpen_Settings,
    "CFZ_MIOpen_Paths":           CFZ_MIOpen_Paths,
    "CFZ_MIOpen_DBInfo":          CFZ_MIOpen_DBInfo,
    "CFZ_MIOpen_Solvers":         CFZ_MIOpen_Solvers,
    "CFZ_MIOpen_SolversFallback": CFZ_MIOpen_SolversFallback,
    "CFZ_MIOpen_SolversBuild":    CFZ_MIOpen_SolversBuild,
    "CFZ_MIOpen_SolversDirectASM":CFZ_MIOpen_SolversDirectASM,
    "CFZ_MIOpen_SolversDirectOCL":CFZ_MIOpen_SolversDirectOCL,
    "CFZ_MIOpen_SolversWinograd": CFZ_MIOpen_SolversWinograd,
    "CFZ_MIOpen_SolversIGEMM":    CFZ_MIOpen_SolversIGEMM,
    "CFZ_MIOpen_SolversCK":       CFZ_MIOpen_SolversCK,
    "CFZ_CuDNN":                  CFZ_CuDNN,
    "CFZ_CuDNN_Benchmark":        CFZ_CuDNN_Benchmark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFZ_save_conditioning":      "CFZ Save Conditioning",
    "CFZ_load_conditioning":      "CFZ Load Conditioning",
    "CFZ_PrintMarker":            "CFZ Print Marker",
    "CFZ_MIOpen_Settings":        "CTD's MIOpen Settings",
    "CFZ_MIOpen_Paths":           "CTD's MIOpen Paths",
    "CFZ_MIOpen_DBInfo":          "CTD's MIOpen DB Info",
    "CFZ_MIOpen_Solvers":         "CTD's MIOpen Solvers — Algorithms",
    "CFZ_MIOpen_SolversFallback": "CTD's MIOpen Solvers — Fallback",
    "CFZ_MIOpen_SolversBuild":    "CTD's MIOpen Solvers — Build Methods",
    "CFZ_MIOpen_SolversDirectASM":"CTD's MIOpen Solvers — Direct ASM",
    "CFZ_MIOpen_SolversDirectOCL":"CTD's MIOpen Solvers — Direct OCL",
    "CFZ_MIOpen_SolversWinograd": "CTD's MIOpen Solvers — Winograd",
    "CFZ_MIOpen_SolversIGEMM":    "CTD's MIOpen Solvers — Implicit GEMM",
    "CFZ_MIOpen_SolversCK":       "CTD's MIOpen Solvers — Group Conv / CK",
    "CFZ_CuDNN":                  "CTD's CuDNN Settings",
    "CFZ_CuDNN_Benchmark":        "CTD's CuDNN Benchmark",
}