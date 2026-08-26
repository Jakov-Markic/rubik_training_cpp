cmake_minimum_required(VERSION 3.21)
project(rubik_training_cpp CXX)

set(CMAKE_CXX_STANDARD 20)  # LibTorch 2.13's headers require C++20 (designated initializers, bit-field defaults)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# All executables land in build/ regardless of which subdirectory (e.g. model/) defines
# them, so every tool sits next to the DLL-copy targets' single shared destination.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# --- OpenCV: prebuilt Windows package vendored under third_party/opencv -------------
set(RUBIK_OPENCV_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv/build" CACHE PATH "OpenCV build dir")
find_package(OpenCV REQUIRED PATHS "${RUBIK_OPENCV_DIR}" NO_DEFAULT_PATH)
set(RUBIK_OPENCV_BIN_DIR "${RUBIK_OPENCV_DIR}/x64/vc16/bin")

# --- Optional LibTorch -------------------------------------------------------------
# rubik_training/venv has torch==2.5.1+cu121 (used by the Python scripts), but its
# bundled Caffe2Config.cmake unconditionally requires a full CUDA Toolkit to be
# findable by CMake (which isn't installed here) even for a CPU-only C++ build target.
# Rather than install a multi-GB CUDA Toolkit just to satisfy find_package(CUDA), we
# vendor a separate CPU-only torch wheel (third_party/torch_cpu/, installed via
# `pip install --target ... --index-url https://download.pytorch.org/whl/cpu torch
# --no-deps`) purely for its LibTorch headers/libs/cmake package -- it never touches
# the venv's GPU torch install. TorchScript files exported by the Python (cu121) side
# still load fine here; TorchScript is forward/backward compatible across versions.
option(RUBIK_BUILD_TORCH_TOOLS "Build LibTorch-dependent tools (pseudo-label inference, training)" OFF)
set(RUBIK_TORCH_CMAKE_PREFIX
    "${CMAKE_SOURCE_DIR}/third_party/torch_cpu/torch/share/cmake"
    CACHE PATH "Torch cmake package dir (torch.utils.cmake_prefix_path)")

# --- common: shared JSON/YOLO-label/image/geometry helpers used by every tool ------
add_library(rubik_common STATIC
    common/src/coco.cpp
    common/src/yolo_labels.cpp
    common/src/image_io.cpp
    common/src/geometry.cpp
)
target_include_directories(rubik_common PUBLIC
    common/include
    third_party
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(rubik_common PUBLIC ${OpenCV_LIBS})

# All tool executables land in the same output directory, so copying runtime DLLs as a
# POST_BUILD step on each target races Ninja's parallel jobs against each other (multiple
# targets' Copy-Item hitting the same destination file at once -> "in use" errors). Do it
# once via a single custom target that every tool depends on instead.
add_custom_target(rubik_copy_opencv_dlls
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}"
    COMMAND powershell -NoProfile -Command
        "Copy-Item -Path '${RUBIK_OPENCV_BIN_DIR}/*.dll' -Destination '${CMAKE_BINARY_DIR}' -Force"
    VERBATIM
)

function(rubik_add_tool name)
    add_executable(${name} tools/${name}/main.cpp)
    target_link_libraries(${name} PRIVATE rubik_common)
    add_dependencies(${name} rubik_copy_opencv_dlls)
endfunction()

rubik_add_tool(extract_video_frames)
rubik_add_tool(dedupe_similar_frames)
rubik_add_tool(yolo_pose_to_coco_keypoints)
rubik_add_tool(coco_keypoints_to_yolo_pose)
rubik_add_tool(merge_coco_keypoints_datasets)
rubik_add_tool(check_keypoint_label_consistency)
rubik_add_tool(visualize_keypoint_labels)

if(RUBIK_BUILD_TORCH_TOOLS)
    # Caffe2's cuda.cmake references CUDA::nvToolsExt (profiling markers) unconditionally
    # when building against a CUDA-enabled torch. Newer CMake's CUDAToolkit module doesn't
    # always create that legacy imported target for this CUDA/CMake version combination
    # (NVTX3 vs. old NVTX packaging churn) -- stand in a no-op so the link-interface
    # reference resolves. We don't use NVTX profiling markers, so this is a correctness
    # no-op, not a missing-feature workaround.
    if(NOT TARGET CUDA::nvToolsExt)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    endif()

    find_package(Torch REQUIRED PATHS "${RUBIK_TORCH_CMAKE_PREFIX}" NO_DEFAULT_PATH)
    # <torch root>/share/cmake -> <torch root>/lib holds the DLLs the pip wheel bundles.
    get_filename_component(RUBIK_TORCH_PY_ROOT "${RUBIK_TORCH_CMAKE_PREFIX}/../.." ABSOLUTE)
    set(RUBIK_TORCH_BIN_DIR "${RUBIK_TORCH_PY_ROOT}/lib")

    add_custom_target(rubik_copy_torch_dlls
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}"
        COMMAND powershell -NoProfile -Command
            "Copy-Item -Path '${RUBIK_TORCH_BIN_DIR}/*.dll' -Destination '${CMAKE_BINARY_DIR}' -Force"
        VERBATIM
    )

    add_executable(pseudo_label_real_cube_images tools/pseudo_label_real_cube_images/main.cpp)
    target_link_libraries(pseudo_label_real_cube_images PRIVATE rubik_common "${TORCH_LIBRARIES}")
    target_include_directories(pseudo_label_real_cube_images PRIVATE ${TORCH_INCLUDE_DIRS})
    add_dependencies(pseudo_label_real_cube_images rubik_copy_opencv_dlls rubik_copy_torch_dlls)

    add_subdirectory(model)
endif()
