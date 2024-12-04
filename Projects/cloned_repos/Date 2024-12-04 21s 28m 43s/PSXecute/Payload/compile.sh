#!/bin/bash
set -e
echo "[*] Compiling to bitcode..."
clang -target i686-w64-mingw32 -emit-llvm -c src/main.c -o main.bc
echo "[*] Applying llvm passes..."
opt -load-pass-plugin=../Transpiler/passes/libremove_dll_import.so \
    -load-pass-plugin=../Transpiler/passes/libdll_call_transform.so \
    -load-pass-plugin=../Transpiler/passes/libchange_triple.so \
    -load-pass-plugin=../Transpiler/passes/libchange_metadata.so \
    -passes="dll-call-transform,remove-dllimport,change-triple,change-metadata" \
    main.bc -o output.bc
echo "[*] Compiling and linking..."
llc -filetype=obj -o output.o output.bc
/home/me/x-tools/mipsel-unknown-linux-gnu/bin/mipsel-unknown-linux-gnu-gcc \
    -nostdlib -fvisibility=hidden -march=r3000 \
    -msoft-float -mabi=32 -mips1 -mno-abicalls -mlong-calls -G0 \
    -T aux/linker.ld -o payload output.o
echo "[*] Extractng shellcode..."
python3 ./aux/extract.py payload
echo "[*] Dumping shellcode..."
xxd -i payload.bin
