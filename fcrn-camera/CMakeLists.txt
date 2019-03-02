file(GLOB fcrnCameraSources *.cpp)
file(GLOB fcrnCameraIncludes *.h )

cuda_add_executable(fcrn-camera ${fcrnCameraSources})

target_link_libraries(fcrn-camera nvcaffe_parser nvinfer jetson-inference)

install(TARGETS fcrn-camera DESTINATION bin)
