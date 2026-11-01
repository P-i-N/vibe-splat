find_package(Catch2 3 CONFIG REQUIRED)

add_executable(splat_tests
  test_support.cpp
  test_selftest.cpp
  test_radix_sort.cpp
  test_ply.cpp
  test_render.cpp
)
target_link_libraries(splat_tests PRIVATE splat_core Catch2::Catch2WithMain)
splat_target_defaults(splat_tests)

include(Catch)
catch_discover_tests(splat_tests)

# End-to-end check of the shipped binary.
add_test(NAME cli_selftest COMMAND splat selftest)
add_test(NAME cli_help COMMAND splat --help)
add_test(NAME cli_make_test_scene COMMAND splat make-test-scene "${CMAKE_CURRENT_BINARY_DIR}/cli_scene.ply" --count 5000)
add_test(NAME cli_render COMMAND splat render "${CMAKE_CURRENT_BINARY_DIR}/cli_scene.ply"
         -o "${CMAKE_CURRENT_BINARY_DIR}/cli_render.png" --width 320 --height 180)
set_tests_properties(cli_render PROPERTIES DEPENDS cli_make_test_scene)
