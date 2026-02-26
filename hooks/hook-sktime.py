from importlib.util import find_spec

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# Legacy/optional hook: only relevant when packaging sktime reference code paths.
datas = copy_metadata("sktime") + collect_data_files("sktime")

hiddenimports = [
    module
    for module in collect_submodules("sktime")
    if find_spec(module) is not None
]
