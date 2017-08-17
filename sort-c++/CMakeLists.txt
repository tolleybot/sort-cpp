cmake_minimum_required(VERSION 2.8)
project(sort_test)

find_package(OpenCV 3 REQUIRED)

set(SOURCE_FILES Sort.cpp
        Sort.h
        main.cpp
        KalmanTracker.h
        KalmanTracker.cpp
        Hungarian.h
        Hungarian.cpp
        )

set(INCLUDE_DIR /usr/local/include
        )

include_directories(${INCLUDE_DIR})

link_directories(/usr/lib
        /usr/local/lib)

add_executable(sort_test ${SOURCE_FILES})


set_property(TARGET sort_test PROPERTY CXX_STANDARD 11)

target_link_libraries(sort_test
        ${OpenCV_LIBS}
        )

