FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  GIT_SHALLOW TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(qoraphasecu_tests
  test_cuda.cpp
  test_operators.cpp
  test_problem.cpp
  test_recipe.cpp
  test_recover.cpp)
target_include_directories(qoraphasecu_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(qoraphasecu_tests PRIVATE qoraphasecu qoraphasecu_warnings GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(qoraphasecu_tests DISCOVERY_TIMEOUT 60)
