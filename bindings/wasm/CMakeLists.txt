if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "bindings/wasm must be configured with emcmake (Emscripten)")
endif()

add_executable(ds_wasm src/ds_wasm.cpp)
target_link_libraries(ds_wasm PRIVATE directedstructure_core)
target_compile_options(ds_wasm PRIVATE -O3)

set(DS_WASM_EXPORTS
    _ds_last_error
    _ds_session_create _ds_session_step _ds_session_progress
    _ds_session_current _ds_session_best
    _ds_session_finish _ds_session_scores _ds_session_coincidence _ds_session_free
    _ds_debug_rng64
    _malloc _free)
list(JOIN DS_WASM_EXPORTS "," _exports)

target_link_options(ds_wasm PRIVATE
    -fexceptions
    -sALLOW_MEMORY_GROWTH=1
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createDsCore
    -sSINGLE_FILE=1
    -sENVIRONMENT=web,worker,node
    -sEXPORTED_FUNCTIONS=${_exports}
    -sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToNewUTF8,HEAP32,HEAPF64
    -sFILESYSTEM=0)

set_target_properties(ds_wasm PROPERTIES
    OUTPUT_NAME ds_core
    SUFFIX ".mjs"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/js/dist)
