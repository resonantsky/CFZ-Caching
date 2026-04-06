import warnings
warnings.filterwarnings("ignore", message="Should have tb<=t1", module="torchsde")

from .cfz_caching_condition import save_conditioning, load_conditioning, CFZ_PrintMarker
from .cfz_miopen import CFZ_MIOpen_Profile, CFZ_MIOpen_Settings, CFZ_MIOpen_Solvers, CFZ_MIOpen_Paths, CFZ_MIOpen_DBInfo

NODE_CLASS_MAPPINGS = {
    "CFZ_save_conditioning": save_conditioning,
    "CFZ_load_conditioning": load_conditioning,
    "CFZ_PrintMarker": CFZ_PrintMarker,
    "CFZ_MIOpen_Profile": CFZ_MIOpen_Profile,
    "CFZ_MIOpen_Settings": CFZ_MIOpen_Settings,
    "CFZ_MIOpen_Solvers": CFZ_MIOpen_Solvers,
    "CFZ_MIOpen_Paths": CFZ_MIOpen_Paths,
    "CFZ_MIOpen_DBInfo": CFZ_MIOpen_DBInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFZ_save_conditioning": "CeeTeeDee's Save Conditioning",
    "CFZ_load_conditioning": "CeeTeeDee's Load Conditioning",
    "CFZ_PrintMarker": "CeeTeeDee's Print Marker",
    "CFZ_MIOpen_Profile": "CeeTeeDee's MIOpen Profile",
    "CFZ_MIOpen_Settings": "CeeTeeDee's MIOpen Settings",
    "CFZ_MIOpen_Solvers": "CeeTeeDee's MIOpen Solvers",
    "CFZ_MIOpen_Paths": "CeeTeeDee's MIOpen Paths",
    "CFZ_MIOpen_DBInfo": "CeeTeeDee's MIOpen DB Info",
}