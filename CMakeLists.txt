cmake_minimum_required(VERSION 3.10)
project(AutoAlgorama VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif(APPLE)
    # macOS-specific settings
    set(CMAKE_MACOSX_RPATH ON)
endif()

# Find Eigen (header-only library)
find_package(Eigen3 REQUIRED)

# Find SDL2
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)

# ImGui source files
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui-1.91.9b)

set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

include_directories(./include)

add_executable(${PROJECT_NAME}
        exec/auto_algorama.cpp
        src/windows/main_window.cpp
        ${IMGUI_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${SDL2_INCLUDE_DIRS}
)

# Platform-specific linking
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        Eigen3::Eigen
        ${SDL2_LIBRARIES}
        OpenGL::GL
    )
elseif(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        Eigen3::Eigen
        ${SDL2_LIBRARIES}
        OpenGL::GL
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
else()
    # Linux
    target_link_libraries(${PROJECT_NAME} PRIVATE
        Eigen3::Eigen
        ${SDL2_LIBRARIES}
        OpenGL::GL
        dl
    )
endif()

# Add basic test (if test files exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    add_subdirectory(tests)
endif()
