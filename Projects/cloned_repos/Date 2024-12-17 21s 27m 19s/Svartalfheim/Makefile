#Makefile 

CC_X64	= x86_64-w64-mingw32-gcc

CFLAGS = -m64 -falign-jumps=1 -falign-labels=1 -nostdlib -ffunction-sections
CFLAGS := $(CFLAGS) -fpack-struct=8 -fno-ident -fno-asynchronous-unwind-tables
CFLAGS := $(CFLAGS) -Wconversion -Os -Wl,-e,EntryPoint -fPIC -Wl,--image-base=0
CFLAGS := $(CFLAGS) -Wl,-s,--no-seh -w -masm=intel

INCLUDE			= -I include
OUT_EXE			= bin/temp.x64.exe
OUT_SHELLCODE	= bin/shellcode.bin

.SILENT: x64

x64:
	@ nasm -f win64 src/asm/Entry.s -o bin/entry.o
	@ nasm -f win64 src/asm/Utils.s -o bin/utils.o
	@ nasm -f win64 src/asm/End.s -o bin/end.o
	@ $(CC_X64) src/*.c  bin/*.o $(INCLUDE) $(CFLAGS) -o $(OUT_EXE) 
	@ python3 extract.py -f $(OUT_EXE) -o $(OUT_SHELLCODE)
	@ rm bin/*.o
	@ rm $(OUT_EXE)
	@ echo "[*] Shellcode ready in $(OUT_SHELLCODE)"


clean:
	@ rm bin/*