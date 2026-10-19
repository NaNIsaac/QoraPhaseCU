# Writes INPUT as a byte array NAME (plus NAME_size) into the C++ source OUTPUT.
file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" length)
math(EXPR size "${length} / 2")
string(REGEX REPLACE "(..)" "0x\\1," bytes "${hex}")
file(WRITE "${OUTPUT}"
  "#include <cstddef>\n"
  "extern const unsigned char ${NAME}[];\n"
  "extern const std::size_t ${NAME}_size;\n"
  "alignas(8) const unsigned char ${NAME}[] = {${bytes}};\n"
  "const std::size_t ${NAME}_size = ${size};\n")
