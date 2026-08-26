@echo off
REM GPU-enabled configure: builds against the venv's CUDA torch (cu121) instead of the
REM CPU-only vendored copy. Requires the CUDA 12.1 Toolkit installed (nvcc + cudart, no
REM display driver -- see GUIDE_LIBTORCH_TRAINING.md). Bakes in three workarounds needed
REM to get CMake's CUDA detection working on this machine -- see README_CPP.md's
REM "GPU build" section for what each one is for and why it's safe here:
REM   1. Short (8.3) paths for the CUDA Toolkit dir, to dodge a CMake/FindCUDA.cmake
REM      backslash-escaping bug (CMP0219) when the path contains spaces.
REM   2. -allow-unsupported-compiler: nvcc 12.1 predates this MSVC version's release and
REM      refuses to run against it otherwise. We don't compile any .cu files ourselves
REM      (LibTorch's CUDA kernels are precompiled), so nvcc's actual codegen is never
REM      exercised -- this only unblocks CMake's compiler-detection step.
REM   3. _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH: Microsoft's own official escape hatch
REM      for the matching STL-side version check (STL1002).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set VENV=c:\Users\User\Desktop\SolveMeCube\rubik_training\venv\Scripts
set TORCH_CMAKE=c:\Users\User\Desktop\SolveMeCube\rubik_training\venv\Lib\site-packages\torch\share\cmake
set CUDA_ROOT=C:/Progra~1/NVIDIA~2/CUDA/v12.1
"%VENV%\cmake.exe" -B build -G Ninja -DCMAKE_MAKE_PROGRAM="%VENV%\ninja.exe" -DCMAKE_BUILD_TYPE=Release ^
  -DRUBIK_BUILD_TORCH_TOOLS=ON ^
  -DRUBIK_TORCH_CMAKE_PREFIX="%TORCH_CMAKE%" ^
  -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_ROOT% ^
  -DCMAKE_CUDA_COMPILER=%CUDA_ROOT%/bin/nvcc.exe ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" ^
  %*
