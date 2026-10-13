# CS7000 — coperception local setup log (macOS, Apple Silicon)

My working notes for getting the [coperception](https://github.com/coperception/coperception)
codebase running locally so I can preprocess the V2X-Sim 2.0 LiDAR subset. Everything here is
CPU-only preprocessing on my laptop; GPU model evaluation happens later on Colab.

## Machine and layout

| Item | Value |
| --- | --- |
| Machine | MacBook Air M4, macOS (Darwin 25.6.0), arm64 |
| Python | 3.11.17 |
| Repo | `/Users/ganeshadhikari/Desktop/CS7000/dataset/coperception` |
| Virtualenv | `<repo>/venv` |
| Raw dataset | `/Users/ganeshadhikari/Desktop/CS7000/dataset/extracted` |

Starting point before these notes: repo cloned, venv created and activated, requirements installed
with `mmcv` removed, package installed editable via `pip install -e . --no-deps`, dataset extracted.

---

## 1. Pin setuptools below 81

coperception's mapping module imports `pkg_resources`, which setuptools 81 removed.

```bash
pip install "setuptools<81"
```

Already satisfied — the venv had **80.10.2**. I confirmed `pkg_resources` still imports:

```bash
python -c "import pkg_resources; print(pkg_resources.__file__)"
```

It imported but printed the deprecation warning. Rather than chase the warning with a further
downgrade, I checked who actually uses it:

```bash
grep -rn "pkg_resources" coperception tools --include="*.py"
```

Only one hit, `coperception/utils/mapping/mapping.py`. Since step 2 rewrites that file anyway, the
`pkg_resources` import disappears entirely and the warning goes with it.

---

## 2. Build the `mapping` native extension for macOS arm64

The repo only ships prebuilt binaries for Linux x86_64 and Windows:

```
mapping.cp37-win_amd64.pyd
mapping.cpython-37m-x86_64-linux-gnu.so
```

On Darwin the bootstrap raises `NotImplementedError: Your system : Darwin, is currently not
supported.` So the extension has to be compiled from source.

### 2.1 Build tools

```bash
brew install cmake eigen
pip install pybind11
```

Versions I ended up with: CMake 4.4.3, Eigen 5.0.1, pybind11 3.1.0.

### 2.2 First configure attempt

```bash
cd coperception/utils/mapping/dynamic_library
mkdir -p build && cd build
cmake .. \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
  -DPYTHON_EXECUTABLE=$(which python)
```

**Error 1 — CMake minimum version:**

```
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

CMake 4.x dropped support for `cmake_minimum_required(VERSION 3.4)`. I bumped the floor in
`dynamic_library/CMakeLists.txt`:

```cmake
# before
cmake_minimum_required(VERSION 3.4)
# after
cmake_minimum_required(VERSION 3.15)
```

**Error 2 — Eigen target not found:**

```
CMake Error at CMakeLists.txt:12 (target_link_libraries):
  Target "mapping" links to:
    Eigen3::Eigen
  but the target was not found.
```

The cause is a version mismatch that failed silently. The original line was

```cmake
find_package(Eigen3 3.2.7 QUIET CONFIG)
```

Homebrew installs Eigen **5.0.1**, and Eigen's `Eigen3ConfigVersion.cmake` treats a single requested
version `3.2.7` as matching only `3.2.7.x`. Because the call is `QUIET` and not `REQUIRED`, the
failure produced no message — it just never defined the `Eigen3::Eigen` target, and the link step
blew up later. Eigen 5 also needs at least C++14, so I raised the standard at the same time:

```cmake
# before
set(CMAKE_CXX_STANDARD 11)
find_package(Eigen3 3.2.7 QUIET CONFIG)

# after
set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED CONFIG)
```

### 2.3 Successful build

```bash
cd coperception/utils/mapping/dynamic_library
rm -rf build && mkdir -p build && cd build
cmake .. \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
  -DPYTHON_EXECUTABLE=$(which python)
cmake --build . --config MinSizeRel -j4
```

```
[ 50%] Building CXX object CMakeFiles/mapping.dir/mapping.cpp.o
[100%] Linking CXX shared module mapping.cpython-311-darwin.so
[100%] Built target mapping
```

Architecture check, then install it next to `mapping.py`:

```bash
file mapping.cpython-311-darwin.so
# Mach-O 64-bit bundle arm64
lipo -archs mapping.cpython-311-darwin.so
# arm64

cp mapping.cpython-311-darwin.so ../../
```

The `cpython-311` ABI tag and `arm64` slice both match this venv's interpreter.

### 2.4 Rewrite `mapping.py`

The original used `platform.system()` sniffing, `pkg_resources.resource_filename`, and
`imp.load_dynamic` — `imp` was removed in Python 3.12 and the OS table has no Darwin entry. I
replaced the whole file so it discovers the right binary from the interpreter's own extension
suffixes and loads it with `importlib`:

```python
import importlib.machinery
import importlib.util
import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_native_mapping():
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = os.path.join(_MODULE_DIR, "mapping" + suffix)
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location(__name__, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        globals().update(
            {k: v for k, v in vars(module).items() if not k.startswith("__")}
        )
        return

    raise ImportError(
        "No mapping extension module matching this interpreter was found in "
        f"{_MODULE_DIR} (tried suffixes: "
        f"{', '.join(importlib.machinery.EXTENSION_SUFFIXES)}). Build it from "
        "coperception/utils/mapping/dynamic_library with CMake and copy the "
        "resulting library here."
    )


_load_native_mapping()
```

`EXTENSION_SUFFIXES` is `['.cpython-311-darwin.so', '.abi3.so', '.so']` here, so it picks up the new
binary first and ignores the stale Linux/Windows ones. This also keeps working on Linux and Windows
without any OS branching, and drops both `imp` and `pkg_resources`.

`coperception/utils/__init__.py` does `from .mapping import mapping`, and callers use
`mapping.compute_logodds_dp(...)`, so copying the symbols into module globals keeps that call path
intact.

### 2.5 Missing runtime dependencies

Importing the module pulled in `nuscenes-devkit`, which surfaced four missing packages one at a
time (`sklearn`, then `cachetools`, plus devkit extras). Installed in two batches:

```bash
pip install scikit-learn tqdm pycocotools
pip install cachetools pyquaternion descartes Pillow fire parameterized
```

pip warns that nuscenes-devkit 1.2.0 wants `numpy<2.0.0` and `Shapely~=2.0.3` while I have numpy
2.4.6 / shapely 2.1.2. I left these as-is — the metadata pin is conservative, numba 0.68 supports
numpy 2.x, and the dataset loads and runs fine (verified in steps 4 and 5). Worth revisiting if
anything numeric looks wrong downstream.

---

## 3. Three compatibility patches

I checked each file before editing. **All three were already patched**, so I changed nothing.

| File | Expected patch | State |
| --- | --- | --- |
| `coperception/utils/mean_ap.py` | local `print_log` instead of `mmcv.utils` | already done |
| `coperception/utils/convolutional_rnn/utils.py` | `collections.abc.Iterable` | already done |
| `tools/det/test_codet.py` | capture `eval_results` from `eval_map()` | already done |

Verification commands and what they returned:

```bash
grep -n "mmcv\|def print_log" coperception/utils/mean_ap.py
# 3:def print_log(msg, logger=None):      <- no mmcv import present

grep -n "collections" coperception/utils/convolutional_rnn/utils.py
# 1:import collections
# 10:        if isinstance(x, collections.abc.Iterable):

grep -n "eval_map(" tools/det/test_codet.py
# 426:        mean_ap, eval_results = eval_map(
# 437:        mean_ap, eval_results = eval_map(
# 450:    mean_ap_local_average, eval_results_average = eval_map(
# 460:    mean_ap_local_average, eval_results_average = eval_map(
```

`mean_ap.py` lines 3-4 for reference:

```python
def print_log(msg, logger=None):
    print(msg)
```

---

## 4. Verification

Ran with deprecation and user warnings promoted to errors, so anything noisy fails loudly. I also
smoke-tested the compiled extension rather than only importing it:

```bash
python -W error::DeprecationWarning -W error::UserWarning -c "
import coperception
from coperception.utils.mean_ap import eval_map
from coperception.utils import mapping
import numpy as np

print('coperception :', coperception.__file__)
print('eval_map     :', eval_map)

pts      = np.array([[1,0,0],[2,0,0],[3,1,0],[4,-1,0.5]], dtype=np.float32)
origins  = np.zeros((1,3), dtype=np.float32)
pc_range = np.array([-10,-10,-5,10,10,5], dtype=np.float32)
out = np.asarray(mapping.compute_logodds_dp(pts, origins, pc_range, range(pts.shape[0]), 0.25))
print('logodds out  : shape', out.shape, '| nonzero', int(np.count_nonzero(out)))
"
```

Output:

```
coperception : .../coperception/coperception/__init__.py
eval_map     : <function eval_map at 0x1218ca020>
logodds out  : shape (160000,) | nonzero 76
```

No errors, no warnings, and the ray-casting actually produced 76 non-zero log-odds cells, so the
native library is doing real work and not just loading.

Note on the signature — `indices` is a sequence of ints and `voxel_size` is a scalar float, not
arrays. I got a `TypeError` the first time by passing numpy arrays for both.

---

## 5. `create_data_det.py` arguments and dataset check

### 5.1 Command-line arguments

From the argparse block at `tools/det/create_data_det.py:586-614`:

| Flag | Short | Type | Default | Meaning |
| --- | --- | --- | --- | --- |
| `--root` | `-r` | str | `/mnt/NAS/.../V2X-Sim-1.0-raw` | Root path to the dataset |
| `--current_agent` | `-c` | int | `0` | Parsed but unused — the loop uses `from_agent`/`to_agent` |
| `--scene_begin` | `-b` | int | `0` | First scene index (inclusive) |
| `--scene_end` | `-e` | int | `100` | Last scene index (exclusive) |
| `--savepath` | `-p` | str | `/mnt/NAS/.../V2X-Sim-1.0-trainval/` | Output directory |
| `--from_agent` | — | int | `0` | First agent index (inclusive) |
| `--to_agent` | — | int | `6` | Last agent index (exclusive) |
| `--dataset_version` | — | str | `v2.0` | `v2.0` or `v2.0-mini` |

Two behaviours worth remembering:

- `--dataset_version v2.0-mini` **overrides** `scene_begin`/`scene_end` to `0` and `1`
  (`create_data_det.py:621-623`).
- `--current_agent` is read into `args` but never used; agent iteration is
  `for current_agent in range(args.from_agent, args.to_agent)` at line 634.

**The script must be run from `tools/det/`.** Line 629 hardcodes a relative path:

```python
scene_files_loc = '../../coperception/utils/v2x_sim_scene_split'
```

Running it from anywhere else fails to find the split files.

### 5.2 Dataset layout check

```bash
ls extracted
# DATASET_INFO.txt  lidarseg  maps  sweeps  v2.0
```

Loaded it through the same class the script uses, read-only:

```bash
cd coperception/tools/det
../../venv/bin/python -c "
from nuscenes import NuScenes as CoPerceptionDataset
ds = CoPerceptionDataset(version='v2.0',
                         dataroot='/Users/ganeshadhikari/Desktop/CS7000/dataset/extracted',
                         verbose=True)
print('scenes:', len(ds.scene))
print('scene[0] :', ds.scene[0]['name'], ds.scene[0]['nbr_samples'])
print('scene[19]:', ds.scene[19]['name'], ds.scene[19]['nbr_samples'])
"
```

```
20 scene, 2000 sample, 191000 sample_data, 826728 sample_annotation, 130 sensor, 3 map
Done loading in 4.000 seconds.
scenes: 20
scene[0] : scene_1 100
scene[19]: scene_20 100
```

Checks that matter for my run:

- **20 scenes, indices 0-19.** `create_data` builds `res_scenes = range(100)` then slices
  `res_scenes[scene_begin:scene_end]` and indexes `coperception_dataset.scene[scene_idx]`, so
  `--scene_begin 0 --scene_end 20` maps exactly onto the 20 scenes I have. Anything past 19 would
  be an `IndexError`.
- **Every index 0-19 appears in a split.** If a scene index is in none of
  `train/val/test_scenes.txt`, line 42 raises `There is no scene {idx} in the dataset.` My subset is
  fully covered:
  - train: 0, 2, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
  - val: 1, 3, 4
  - test: 5, 8, 19
- **Six agent channels exist**, `LIDAR_TOP_id_0` through `LIDAR_TOP_id_5`, 2000 `.pcd.bin` files
  each under `extracted/sweeps/`. Max agents in any scene is 6, so `--to_agent 6` is right.
  Agents per scene actually vary (3 agents x1, 4 x10, 5 x7, 6 x2); the script handles the gaps via
  its `channel_flag` branch, which writes placeholder entries for absent agents.
- First referenced point cloud resolves on disk:
  `sweeps/LIDAR_TOP_id_0/scene_1_000006.pcd.bin`.

Output goes to `<savepath>/<split>/agent<N>/`, where the split is chosen per scene from the split
files — so I get `train/`, `val/` and `test/` subtrees automatically.

---

## 6. Command to run the preprocessing

```bash
cd /Users/ganeshadhikari/Desktop/CS7000/dataset/coperception/tools/det

../../venv/bin/python create_data_det.py \
  --root /Users/ganeshadhikari/Desktop/CS7000/dataset/extracted \
  --scene_begin 0 \
  --scene_end 20 \
  --savepath /Users/ganeshadhikari/Desktop/CS7000/dataset/v2x_sim_20scenes_det \
  --from_agent 0 \
  --to_agent 6 \
  --dataset_version v2.0
```

The `cd` is required, not cosmetic — see the relative `scene_files_loc` note in 5.1.

---

## 7. numpy 2.x validation (scene 0)

pip flags that nuscenes-devkit 1.2.0 declares `numpy<2.0.0` while I have 2.4.6, so before
committing to the full run I validated a partial scene-0 output: **373 `.npy` files** across agents
0-3, inspected for NaNs, dtype drift, shape errors and unexpected empties.

Result: **clean.**

| Check | Result |
| --- | --- |
| Files scanned | 373 (all real dicts, no placeholders) |
| NaN / Inf | 0 / 0 |
| `voxel_indices_*` round-trips | 1119 / 1119 OK |
| Index bounds vs `map_dims` [256,256,13] | all in range |
| Unexpectedly empty arrays | 0 |
| Occupied voxels per frame | min 5506, median 11467, max 16727 |
| Sparse reg targets per frame | min 1848, median 4758, max 7494 |
| Frames with >0 targets | 373 / 373 |

Two things looked wrong at first and turned out to be by design:

- **`reg_target_sparse` is float64, not float32.** I expected float32 (the `box_data` buffer at
  line 192 is float32), but `reg_target` is allocated at `obj_util.py:412` with a bare
  `np.zeros((...))` and no `dtype`, which is float64 on *every* numpy version. The reconstruction
  side at `obj_util.py:756` allocates the same way, so producer and consumer agree. Not version
  related.
- **`vis_occupy_indices` / `vis_free_indices` are empty `(4,0)` for agent 0.** Agent 0 is the
  roadside unit, and `create_data_det.py:365-366` short-circuits it: when `config.is_cross_road`
  is true it appends `np.zeros([256,256,13])` instead of ray-casting, so `np.where(>0)` and
  `np.where(<0)` are both empty. Agents 1-3 had non-empty visibility on every frame.

The script's own round-trip asserts (`create_data_det.py:552-580`) also passed on all 373 frames —
they re-derive the voxel grid, labels, regression targets and visibility maps from what is about
to be saved and compare. Silent numeric corruption from an ABI mismatch would trip those.

Conclusion: numpy 2.4.6 causes no problem for this pipeline. The temp output was deleted after
inspection.

---

## Summary of files I changed

| File | Change |
| --- | --- |
| `coperception/utils/mapping/dynamic_library/CMakeLists.txt` | CMake floor 3.4 -> 3.15; C++11 -> C++17; `find_package(Eigen3 3.2.7 QUIET CONFIG)` -> `find_package(Eigen3 REQUIRED CONFIG)` |
| `coperception/utils/mapping/mapping.py` | Rewrote to load the extension via `importlib`; dropped `imp`, `pkg_resources` and OS sniffing |
| `coperception/utils/mapping/mapping.cpython-311-darwin.so` | New — compiled arm64 build output |

Packages added to the venv: `scikit-learn`, `tqdm`, `pycocotools`, `cachetools`, `pyquaternion`,
`descartes`, `Pillow`, `fire`, `parameterized`.

## Open items

- ~~numpy 2.4.6 and shapely 2.1.2 violate nuscenes-devkit's declared pins.~~ **Resolved** — see
  section 7. The pins are conservative metadata; the output is clean.
- The stale `mapping.cp37-win_amd64.pyd` and `mapping.cpython-37m-x86_64-linux-gnu.so` are still in
  the tree. Harmless — the new loader skips them — and worth keeping for Colab, which is Linux
  x86_64 but on Python 3.11, so that old `cp37` binary will not load there either. I will need to
  rebuild the extension on Colab the same way.
