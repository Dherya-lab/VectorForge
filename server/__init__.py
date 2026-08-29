import os
import sys

# On Windows, register MinGW DLL directory so that .pyd extension loads seamlessly
if sys.platform == "win32":
    mingw_bin = r"C:\msys64\mingw64\bin"
    if os.path.exists(mingw_bin) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(mingw_bin)
        except Exception:
            pass

# Re-export vectorforge
try:
    import vectorforge
    from vectorforge import VectorForgeIndex, simd_info
except ImportError:
    # If importing directly from the server folder
    sys.path.insert(0, os.path.dirname(__file__))
    import vectorforge
    from vectorforge import VectorForgeIndex, simd_info

__all__ = ["vectorforge", "VectorForgeIndex", "simd_info"]
