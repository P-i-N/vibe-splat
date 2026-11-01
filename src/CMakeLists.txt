add_library(splat_core STATIC
  core/vk_check.cpp
  core/vk_context.cpp
  core/vk_buffer.cpp
  core/vk_compute.cpp
  core/vma_impl.cpp
  core/selftest.cpp
  core/radix_sort.cpp
  core/vk_image.cpp
  splat/gaussian_cloud.cpp
  render/camera.cpp
  render/splat_renderer.cpp
  render/offscreen.cpp
)
target_include_directories(splat_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(splat_core PRIVATE "${STB_INCLUDE_DIR}")
target_link_libraries(splat_core
  PUBLIC volk::volk Vulkan::Headers GPUOpen::VulkanMemoryAllocator glm::glm
  PRIVATE splat_shaders)
target_compile_definitions(splat_core PUBLIC VK_NO_PROTOTYPES)
splat_target_defaults(splat_core)

add_executable(splat cli/main.cpp)
target_link_libraries(splat PRIVATE splat_core)
if(TARGET SDL3::SDL3)
  target_sources(splat PRIVATE viewer/viewer.cpp)
  target_link_libraries(splat PRIVATE SDL3::SDL3)
  target_compile_definitions(splat PRIVATE SPLAT_WITH_VIEWER)
  message(STATUS "Window viewer: enabled (SDL3 ${SDL3_VERSION})")
else()
  message(STATUS "Window viewer: disabled (SDL3 not found)")
endif()
target_compile_definitions(splat PRIVATE SPLAT_VERSION="${PROJECT_VERSION}")
splat_target_defaults(splat)
splat_static_runtime(splat)

install(TARGETS splat RUNTIME DESTINATION bin)
