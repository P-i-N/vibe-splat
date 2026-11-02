find_package(Catch2 3 CONFIG REQUIRED)

add_executable(splat_tests
  test_support.cpp
  test_selftest.cpp
  test_radix_sort.cpp
  test_ply.cpp
  test_cgc.cpp
  test_render.cpp
  test_ingest.cpp
  test_sfm.cpp
  test_train.cpp
  test_densify.cpp
  test_progress.cpp
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
# Compress the test scene to CGC and render the result.
add_test(NAME cli_convert_cgc COMMAND splat convert "${CMAKE_CURRENT_BINARY_DIR}/cli_scene.ply" "${CMAKE_CURRENT_BINARY_DIR}/cli_scene.cgc")
set_tests_properties(cli_convert_cgc PROPERTIES DEPENDS cli_make_test_scene)
add_test(NAME cli_render_cgc COMMAND splat render "${CMAKE_CURRENT_BINARY_DIR}/cli_scene.cgc"
         -o "${CMAKE_CURRENT_BINARY_DIR}/cli_render_cgc.png" --width 320 --height 180)
set_tests_properties(cli_render_cgc PROPERTIES DEPENDS cli_convert_cgc)
# The floater filter options must be accepted (the test scene has nothing to prune).
add_test(NAME cli_convert_prune COMMAND splat convert "${CMAKE_CURRENT_BINARY_DIR}/cli_scene.ply" "${CMAKE_CURRENT_BINARY_DIR}/cli_scene_pruned.ply"
         --prune-sigma 1 --prune-radius 4)
set_tests_properties(cli_convert_prune PROPERTIES DEPENDS cli_make_test_scene)
# Ingesting the image cli_render wrote checks the whole command line path.
add_test(NAME cli_ingest COMMAND splat ingest "${CMAKE_CURRENT_BINARY_DIR}/cli_render.png"
         -o "${CMAKE_CURRENT_BINARY_DIR}/cli_project" --force --focal-35mm 28)
set_tests_properties(cli_ingest PROPERTIES DEPENDS cli_render)
# The whole pipeline from images to poses to a rendered point cloud.
set(SFM_DIR "${CMAKE_CURRENT_BINARY_DIR}/cli_sfm")
add_test(NAME cli_make_test_views COMMAND splat make-test-views "${SFM_DIR}/input" --count 8)
add_test(NAME cli_sfm_ingest COMMAND splat ingest "${SFM_DIR}/input" -o "${SFM_DIR}/project" --force --focal-px 500)
add_test(NAME cli_sfm COMMAND splat sfm "${SFM_DIR}/project")
add_test(NAME cli_sfm_render COMMAND splat render "${SFM_DIR}/project/sparse/0/preview.ply" -o "${SFM_DIR}/preview.png" --width 320 --height 240)
set_tests_properties(cli_sfm_ingest PROPERTIES DEPENDS cli_make_test_views)
set_tests_properties(cli_sfm PROPERTIES DEPENDS cli_sfm_ingest)
set_tests_properties(cli_sfm_render PROPERTIES DEPENDS cli_sfm)
add_test(NAME cli_train COMMAND splat train "${SFM_DIR}/project" --iterations 30)
set_tests_properties(cli_train PROPERTIES DEPENDS cli_sfm)
