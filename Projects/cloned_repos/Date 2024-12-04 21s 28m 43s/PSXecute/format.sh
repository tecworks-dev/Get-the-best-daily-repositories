find . -iname "*.cpp" -o -iname "*.h" -o -iname "*.c" -o -iname "*.hpp" | xargs clang-format -i
