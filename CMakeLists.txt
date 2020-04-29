cmake_minimum_required(VERSION 2.8)
project(LPCNet C)

set(CMAKE_C_STANDARD 99)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -std=c99 -march=native -mfma -O2 -fPIC -fstack-protector-all -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -shared -fPIC -Wl,-z,relro -Wl,-z,noexecstack -Wl,-z,now")

set(CMAKE_SKIP_RPATH true)

include_directories(.)
include_directories(include)
include_directories(src)

add_executable(LPCNet
        include/lpcnet.h
        src/_kiss_fft_guts.h
        src/arch.h
        src/celt_lpc.c
        src/celt_lpc.h
        src/ceps_codebooks.c
        src/common.c
        src/common.h
        src/freq.c
        src/freq.h
        src/kiss_fft.c
        src/kiss_fft.h
        src/lpcnet.c
        src/lpcnet_dec.c
        src/lpcnet_demo.c
        src/lpcnet_enc.c
        src/lpcnet_private.h
        src/nnet.c
        src/nnet.h
        src/nnet_data.c
        src/nnet_data.h
        src/opus_types.h
        src/pitch.c
        src/pitch.h
        src/tansig_table.h
        src/vec.h
        src/vec_avx.h
        config.h
        src/frame_analysis.h
        src/frame_analysis.c
        )

target_link_libraries(LPCNet m)

add_library( # Sets the name of the library.
        lpcnet_native

        # Sets the library as a shared library.
        SHARED

        src/_kiss_fft_guts.h
        src/arch.h
        src/celt_lpc.c
        src/celt_lpc.h
        src/ceps_codebooks.c
        src/common.c
        src/common.h
        src/freq.c
        src/freq.h
        src/kiss_fft.c
        src/kiss_fft.h
        src/lpcnet.c
        src/lpcnet_dec.c
        src/lpcnet_enc.c
        src/lpcnet_private.h
        src/nnet.c
        src/nnet.h
        src/nnet_data.c
        src/nnet_data.h
        src/opus_types.h
        src/pitch.c
        src/pitch.h
        src/tansig_table.h
        src/vec.h
        src/vec_avx.h
        config.h
        src/frame_analysis.h
        src/frame_analysis.c
        )

#target_compile_options(af_matmul PUBLIC -std=c++11)

target_link_libraries(lpcnet_native
        m
        #        pthread
        #        att_af
        #        af_matmul0
        #        ArrayFire::afcpu
        #        boost_system
        )
