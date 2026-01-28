cmake_minimum_required(VERSION 3.10)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()
include(GoogleTest)

# Add test executable
add_executable(AutoAlgorama_tests
    test_functions.cpp
)

target_include_directories(AutoAlgorama_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(AutoAlgorama_tests PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
)

# Discover tests
gtest_discover_tests(AutoAlgorama_tests)
