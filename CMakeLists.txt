cmake_minimum_required(VERSION 3.10)
project(gstreamerTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g -fpermissive")


MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")
MESSAGE(STATUS "CMAKE_SYSTEM_NAME is ${CMAKE_SYSTEM}")
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "current platform: Linux ")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    MESSAGE(STATUS "current platform: Windows")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    MESSAGE(STATUS "current platform: FreeBSD")
ELSE ()
    MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")


if (${CMAKE_SYSTEM} MATCHES "Linux-4.9.140-tegra")
    message("On TEGRA PLATFORM.")
    add_definitions(-DPLATFORM_TEGRA)
    set(SYS_USR_LIB /usr/lib/aarch64-linux-gnu)
    set(SYS_LIB /lib/aarch64-linux-gnu)
else ()
    message("On X86 PLATFORM.")
    set(SYS_USR_LIB /usr/lib/x86_64-linux-gnu)
    set(SYS_LIB /lib/x86_64-linux-gnu)
endif ()

find_package(OpenCV REQUIRED)
set(DS_DIR /opt/nvidia/deepstream/deepstream-5.0)
include_directories(
        /usr/include/gstreamer-1.0
        /usr/include/glib-2.0
        /usr/local/cuda/include
        /home/casit/deeplearning/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6/TensorRT-7.0.0.11/include/
        ${OpenCV_INCLUDE_DIRS}
        ${SYS_USR_LIB}/glib-2.0/include
        ${DS_DIR}/sources/includes
)
link_libraries(
        ${SYS_USR_LIB}/libgstreamer-1.0.so.0
        ${SYS_USR_LIB}/libgobject-2.0.so.0
        ${SYS_USR_LIB}/libglib-2.0.so.0
        ${OpenCV_LIBRARIES}
        /usr/local/cuda/lib64/libcudart.so
        ${SYS_LIB}/libc.so.6
        ${SYS_LIB}/libdl.so.2
        ${SYS_LIB}/libpthread.so.0
        ${SYS_USR_LIB}/libgmodule-2.0.so.0
        ${SYS_LIB}/libm.so.6
        ${SYS_USR_LIB}/libffi.so.6
        ${SYS_LIB}/libpcre.so.3
        ${DS_DIR}/lib/libnvdsgst_meta.so
        ${DS_DIR}/lib/libnvds_meta.so
        ${DS_DIR}/lib/libnvds_inferutils.so
        ${DS_DIR}/lib/libnvds_utils.so
        ${DS_DIR}/lib/libnvds_infercustomparser.so
)

add_executable(gstreamerTest main.cpp)
add_executable(deepstream_infer_tensor_meta_test deepstream_infer_tensor_meta_test.cpp)



