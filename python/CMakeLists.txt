# SKBUILD_SABI_COMPONENT is Development.SABIModule when the wheel targets the stable ABI;
# without it nanobind silently builds against the full API.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core STABLE_ABI NB_STATIC bindings.cpp)
target_link_libraries(_core PRIVATE qoraphasecu qoraphasecu_warnings)
install(TARGETS _core LIBRARY DESTINATION qoraphasecu)
