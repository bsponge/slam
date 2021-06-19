set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ../include)
target_link_directories(${PROJECT_NAME} PUBLIC ../include)
target_link_libraries(${PROJECT_NAME}  GL GLEW GLU Xrandr sfml-graphics sfml-window sfml-system m)

