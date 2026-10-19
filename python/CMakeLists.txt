find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core STABLE_ABI NB_STATIC bindings.cpp)
target_link_libraries(_core PRIVATE qoraphasecu qoraphasecu_warnings)
install(TARGETS _core LIBRARY DESTINATION qoraphasecu)
