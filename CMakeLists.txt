cmake_minimum_required(VERSION 3.14)
project(EyeTorsionTracking CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Include directories
include_directories(include)

# Core source files (exclude main.cpp)
set(CORE_SOURCES
    src/DataLoader.cpp
    src/TorsionDetector.cpp
    src/algorithms/PolarCrossCorrelation.cpp
)

# Core static library
add_library(torsion_core STATIC ${CORE_SOURCES})
target_include_directories(torsion_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(torsion_core PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json)

# Executable target linking core library
add_executable(torsion_app src/main.cpp)
target_link_libraries(torsion_app PRIVATE torsion_core)

# Option to enable diagnostics
option(ENABLE_DIAGNOSTICS "Enable algorithm diagnostic outputs" OFF)
if(ENABLE_DIAGNOSTICS)
    target_compile_definitions(torsion_core PRIVATE DIAGNOSTICS_DEFAULT=true)
    message(STATUS "Build Mode: Diagnostics ENABLED by default")
else()
    target_compile_definitions(torsion_core PRIVATE DIAGNOSTICS_DEFAULT=false)
    message(STATUS "Build Mode: Diagnostics DISABLED by default")
endif()

# -------------------- Testing --------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: prevent overriding parent compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(torsion_tests tests/test_PolarCrossCorrelation.cpp)
target_link_libraries(torsion_tests PRIVATE torsion_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(torsion_tests)
