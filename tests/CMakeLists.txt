find_package(Catch2 3 CONFIG REQUIRED)

add_executable(splat_tests
  test_selftest.cpp
)
target_link_libraries(splat_tests PRIVATE splat_core Catch2::Catch2WithMain)
splat_target_defaults(splat_tests)

include(Catch)
catch_discover_tests(splat_tests)

# End-to-end check of the shipped binary.
add_test(NAME cli_selftest COMMAND splat selftest)
add_test(NAME cli_help COMMAND splat --help)
