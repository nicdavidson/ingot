#!/bin/bash
# Converts .metal shader files into a C source file with embedded strings.
# Usage: ./tools/embed_shaders.sh > build/shader_strings.c

SHADER_DIR="src/compute/shaders"

echo "// Auto-generated — do not edit"
echo "// Embedded Metal shader source strings"
echo ""

for shader in "$SHADER_DIR"/*.metal; do
    name=$(basename "$shader" .metal)
    var_name="shader_${name}_src"
    echo "const char *${var_name} ="
    # Escape backslashes and quotes, wrap each line in quotes
    sed 's/\\/\\\\/g; s/"/\\"/g; s/^/    "/; s/$/\\n"/' "$shader"
    echo ";"
    echo ""
done
