# Upstream imgui's sdl3-binding depends on sdl3 with its default features,
# which on Linux turns on ibus and dbus and pulls libsystemd and libxcrypt
# (an autotools build that needs autoconf-archive). This overlay only changes
# that dependency in vcpkg.json; the build itself is the upstream port.
include("${VCPKG_ROOT_DIR}/ports/imgui/portfile.cmake")
