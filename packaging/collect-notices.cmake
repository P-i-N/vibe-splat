# Gathers the licences of the libraries linked into splat from vcpkg's install
# tree into one file, for release packages.
#
#   cmake -DINSTALLED=<vcpkg_installed>/<triplet> -DOUT=<file> -P collect-notices.cmake
foreach(var INSTALLED OUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "collect-notices.cmake needs -D${var}=...")
  endif()
endforeach()

set(ports sdl3 imgui volk vulkan-headers vulkan-memory-allocator glm stb nlohmann-json eigen3)
set(text "splat is released under the 0BSD licence (see LICENSE). It includes the\nfollowing libraries, under their own licences.\n")
foreach(port IN LISTS ports)
  set(file "${INSTALLED}/share/${port}/copyright")
  if(NOT EXISTS "${file}")
    message(FATAL_ERROR "No licence for ${port} at ${file}")
  endif()
  file(READ "${file}" licence)
  string(APPEND text "\n==============================================================================\n${port}\n==============================================================================\n\n${licence}\n")
endforeach()
file(WRITE "${OUT}" "${text}")
message(STATUS "Wrote ${OUT}")
