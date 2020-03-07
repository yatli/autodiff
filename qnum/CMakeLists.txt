add_executable(qnum test.cpp)
target_include_directories(qnum PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
