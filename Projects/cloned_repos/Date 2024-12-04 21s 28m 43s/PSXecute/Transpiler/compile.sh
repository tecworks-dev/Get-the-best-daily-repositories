#!/bin/bash
echo "[*] Compiling passes"
echo "    remove_dll_import"
clang++ -fPIC -shared ./passes/remove_dll_import.cpp -o ./passes/libremove_dll_import.so $(llvm-config --cxxflags --ldflags --system-libs --libs core)
echo "    dll_call_transform"
clang++ -shared -fPIC ./passes/dll_call_transform.cpp -o ./passes/libdll_call_transform.so $(llvm-config --cxxflags --ldflags --system-libs --libs core passes)
echo "    change_triple"
clang++ -shared -fPIC ./passes/change_triple.cpp -o ./passes/libchange_triple.so $(llvm-config --cxxflags --ldflags --system-libs --libs core passes)
echo "    change_metadata"
clang++ -shared -fPIC ./passes/change_metadata.cpp -o ./passes/libchange_metadata.so $(llvm-config --cxxflags --ldflags --system-libs --libs core passes)
echo "[*] Done"