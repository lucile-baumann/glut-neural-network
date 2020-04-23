project(neural-network)
cmake_minimum_required(VERSION 3.12)

set(OPENGL_LIBRARY_DIR "C:/Program Files/freeglut/lib")
set(GLUT_INCLUDE_DIR "C:/Program Files/freeglut/include")
find_package(GLUT REQUIRED)

add_executable(NeuralNetwork 
main.cc 
reseau.cc)

target_link_libraries(NeuralNetwork ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(NeuralNetwork PRIVATE 
${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR}
)