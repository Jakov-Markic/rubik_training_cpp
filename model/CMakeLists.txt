# Phase 3: from-scratch LibTorch YOLOv8n-pose-shaped model, loss, augmentation,
# dataset loading, and the train/export tools. Only added when
# RUBIK_BUILD_TORCH_TOOLS=ON (see top-level CMakeLists.txt).

add_library(rubik_model STATIC
    src/modules.cpp
    src/yolo_pose_8pt.cpp
    src/augment.cpp
    src/dataset.cpp
    src/loss.cpp
    src/plotting.cpp
    src/evaluate.cpp
)
target_include_directories(rubik_model PUBLIC
    include
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(rubik_model PUBLIC rubik_common "${TORCH_LIBRARIES}")

add_executable(train_cube_pose_8pt tools/train_cube_pose_8pt/main.cpp)
target_link_libraries(train_cube_pose_8pt PRIVATE rubik_model)
add_dependencies(train_cube_pose_8pt rubik_copy_opencv_dlls rubik_copy_torch_dlls)

add_executable(export_cube_pose_8pt tools/export_cube_pose_8pt/main.cpp)
target_link_libraries(export_cube_pose_8pt PRIVATE rubik_model)
add_dependencies(export_cube_pose_8pt rubik_copy_opencv_dlls rubik_copy_torch_dlls)
