# Script mode: packs several .spv files into one xz-compressed C++ source.
# Inputs:
#   GEN_DIR   directory holding <name>.spv for each name
#   NAMES     shader file names joined with ',' (e.g. a.comp,b.vert)
#   TARGET    library name, for the header include
#   TEMPLATE  path of SpirvPack.cpp.in
#   OUTPUT    the .cpp to write
# Each shader becomes a function splat::shaders::<name with '.' -> '_'>().
string(REPLACE "," ";" names "${NAMES}")

set(spvs "")
set(entries "")
set(functions "")
set(offset 0)
set(index 0)
foreach(name IN LISTS names)
  set(spv "${GEN_DIR}/${name}.spv")
  file(SIZE "${spv}" size)
  math(EXPR rem "${size} % 4")
  if(size EQUAL 0 OR NOT rem EQUAL 0)
    message(FATAL_ERROR "${spv} is not a valid SPIR-V module")
  endif()
  math(EXPR words "${size} / 4")
  string(REPLACE "." "_" sym "${name}")
  list(APPEND spvs "${spv}")
  string(APPEND entries "  {${offset}, ${words}}, // ${name}\n")
  string(APPEND functions "std::span<const std::uint32_t> ${sym}() { return shader(${index}); }\n")
  math(EXPR offset "${offset} + ${words}")
  math(EXPR index "${index} + 1")
endforeach()

# cmake -E cat is binary-safe; file(ARCHIVE_CREATE ... FORMAT raw) then
# compresses that single file without any archive headers.
set(bin "${GEN_DIR}/shaders.bin")
execute_process(COMMAND "${CMAKE_COMMAND}" -E cat ${spvs} OUTPUT_FILE "${bin}" COMMAND_ERROR_IS_FATAL ANY)
file(REMOVE "${bin}.xz")
file(ARCHIVE_CREATE OUTPUT "${bin}.xz" PATHS "${bin}" FORMAT raw COMPRESSION XZ COMPRESSION_LEVEL 9)

file(READ "${bin}.xz" hex HEX)
string(REGEX REPLACE "(..)" "0x\\1," bytes "${hex}")
string(REGEX REPLACE "(([^,]*,){16})" "\\1\n  " bytes "${bytes}")
set(PACKED_BYTES "  ${bytes}")
set(UNPACKED_WORDS ${offset})
set(ENTRIES "${entries}")
set(FUNCTIONS "${functions}")
configure_file("${TEMPLATE}" "${OUTPUT}" @ONLY)
