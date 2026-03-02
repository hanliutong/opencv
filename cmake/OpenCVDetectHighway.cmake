# OpenCVDetectHighway.cmake
# Detect Google Highway SIMD library availability

# Highway 源码位于 3rdparty/highway/
# 作为 header-only 使用（只需要头文件路径）

if(WITH_HIGHWAY)
    # 检查 Highway 头文件是否存在
    set(HIGHWAY_INCLUDE_DIR "${OpenCV_SOURCE_DIR}/3rdparty/highway")

    if(EXISTS "${HIGHWAY_INCLUDE_DIR}/hwy/highway.h")
        set(HAVE_HIGHWAY TRUE)
        # 创建 INTERFACE library
        if(NOT TARGET hwy_interface)
            add_library(hwy_interface INTERFACE)
            target_include_directories(hwy_interface INTERFACE "${HIGHWAY_INCLUDE_DIR}")
            # Highway 需要 C++11 或更高
            target_compile_features(hwy_interface INTERFACE cxx_std_11)
        endif()

        # 全局添加 Highway 定义和包含路径，使 intrin.hpp 中的 #if defined(CV_HWY) 生效
        add_definitions(-DCV_HWY=1)
        include_directories("${HIGHWAY_INCLUDE_DIR}")

        message(STATUS "Highway SIMD library found at ${HIGHWAY_INCLUDE_DIR}")
    else()
        set(HAVE_HIGHWAY FALSE)
        message(WARNING "Highway headers not found at ${HIGHWAY_INCLUDE_DIR}/hwy/highway.h")
    endif()
endif()
