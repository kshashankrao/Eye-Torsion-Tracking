cmake_minimum_required(VERSION 3.14)
project(EyeTorsionTracking CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Include directories
include_directories(include)

# Define source files
set(SOURCES
    src/main.cpp
    src/DataLoader.cpp
    src/TorsionDetector.cpp
    src/algorithms/PolarCrossCorrelation.cpp
)

# Executable target
add_executable(torsion_app ${SOURCES})

# Link OpenCV and nlohmann_json
target_link_libraries(torsion_app PRIVATE 
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

# Option to enable diagnostics (toggled via cmake -DENABLE_DIAGNOSTICS=ON)
option(ENABLE_DIAGNOSTICS "Enable algorithm diagnostic outputs" OFF)

if(ENABLE_DIAGNOSTICS)
    target_compile_definitions(torsion_app PRIVATE DIAGNOSTICS_DEFAULT=true)
    message(STATUS "Build Mode: Diagnostics ENABLED by default")
else()
    target_compile_definitions(torsion_app PRIVATE DIAGNOSTICS_DEFAULT=false)
    message(STATUS "Build Mode: Diagnostics DISABLED by default")
endif()
