// Some systems have issues with lowmem address getting corrupted.
// Deal with this before taking a pagetable-related exception.

#define DEVL 1
#include <ntddk.h>
#include <arc.h>
#include "runtime.h"
#include "ppcinst.h"

typedef UCHAR BYTE;
typedef PUCHAR PBYTE;

// Bring in NT types we need:
// LDR_DATA_TABLE_ENTRY
typedef struct _LDR_DATA_TABLE_ENTRY {
    LIST_ENTRY InLoadOrderLinks;
    LIST_ENTRY InMemoryOrderLinks;
    LIST_ENTRY InInitializationOrderLinks;
    PVOID DllBase;
    PVOID EntryPoint;
    ULONG SizeOfImage;
    UNICODE_STRING FullDllName;
    UNICODE_STRING BaseDllName;
    ULONG Flags;
    USHORT LoadCount;
    USHORT TlsIndex;
    union {
        LIST_ENTRY HashLinks;
        struct {
            PVOID SectionPointer;
            ULONG CheckSum;
        };
    };
    ULONG   TimeDateStamp;
} LDR_DATA_TABLE_ENTRY, *PLDR_DATA_TABLE_ENTRY;

// PE header structures:
#include "pshpack4.h"                   // 4 byte packing is the default

#define IMAGE_DOS_SIGNATURE                 0x5A4D      // MZ
#define IMAGE_OS2_SIGNATURE                 0x454E      // NE
#define IMAGE_OS2_SIGNATURE_LE              0x454C      // LE
#define IMAGE_VXD_SIGNATURE                 0x454C      // LE
#define IMAGE_NT_SIGNATURE                  0x00004550  // PE00

#include "pshpack2.h"                   // 16 bit headers are 2 byte packed

typedef struct _IMAGE_DOS_HEADER {      // DOS .EXE header
    USHORT e_magic;                     // Magic number
    USHORT e_cblp;                      // Bytes on last page of file
    USHORT e_cp;                        // Pages in file
    USHORT e_crlc;                      // Relocations
    USHORT e_cparhdr;                   // Size of header in paragraphs
    USHORT e_minalloc;                  // Minimum extra paragraphs needed
    USHORT e_maxalloc;                  // Maximum extra paragraphs needed
    USHORT e_ss;                        // Initial (relative) SS value
    USHORT e_sp;                        // Initial SP value
    USHORT e_csum;                      // Checksum
    USHORT e_ip;                        // Initial IP value
    USHORT e_cs;                        // Initial (relative) CS value
    USHORT e_lfarlc;                    // File address of relocation table
    USHORT e_ovno;                      // Overlay number
    USHORT e_res[4];                    // Reserved words
    USHORT e_oemid;                     // OEM identifier (for e_oeminfo)
    USHORT e_oeminfo;                   // OEM information; e_oemid specific
    USHORT e_res2[10];                  // Reserved words
    LONG   e_lfanew;                    // File address of new exe header
  } IMAGE_DOS_HEADER, *PIMAGE_DOS_HEADER;

#include "poppack.h"                    // Back to 4 byte packing

//
// File header format.
//

typedef struct _IMAGE_FILE_HEADER {
    USHORT  Machine;
    USHORT  NumberOfSections;
    ULONG   TimeDateStamp;
    ULONG   PointerToSymbolTable;
    ULONG   NumberOfSymbols;
    USHORT  SizeOfOptionalHeader;
    USHORT  Characteristics;
} IMAGE_FILE_HEADER, *PIMAGE_FILE_HEADER;

//
// Directory format.
//

typedef struct _IMAGE_DATA_DIRECTORY {
    ULONG   VirtualAddress;
    ULONG   Size;
} IMAGE_DATA_DIRECTORY, *PIMAGE_DATA_DIRECTORY;

#define IMAGE_NUMBEROF_DIRECTORY_ENTRIES    16

//
// Optional header format.
//

typedef struct _IMAGE_OPTIONAL_HEADER {
    //
    // Standard fields.
    //

    USHORT  Magic;
    UCHAR   MajorLinkerVersion;
    UCHAR   MinorLinkerVersion;
    ULONG   SizeOfCode;
    ULONG   SizeOfInitializedData;
    ULONG   SizeOfUninitializedData;
    ULONG   AddressOfEntryPoint;
    ULONG   BaseOfCode;
    ULONG   BaseOfData;

    //
    // NT additional fields.
    //

    ULONG   ImageBase;
    ULONG   SectionAlignment;
    ULONG   FileAlignment;
    USHORT  MajorOperatingSystemVersion;
    USHORT  MinorOperatingSystemVersion;
    USHORT  MajorImageVersion;
    USHORT  MinorImageVersion;
    USHORT  MajorSubsystemVersion;
    USHORT  MinorSubsystemVersion;
    ULONG   Win32VersionValue;
    ULONG   SizeOfImage;
    ULONG   SizeOfHeaders;
    ULONG   CheckSum;
    USHORT  Subsystem;
    USHORT  DllCharacteristics;
    ULONG   SizeOfStackReserve;
    ULONG   SizeOfStackCommit;
    ULONG   SizeOfHeapReserve;
    ULONG   SizeOfHeapCommit;
    ULONG   LoaderFlags;
    ULONG   NumberOfRvaAndSizes;
    IMAGE_DATA_DIRECTORY DataDirectory[IMAGE_NUMBEROF_DIRECTORY_ENTRIES];
} IMAGE_OPTIONAL_HEADER, *PIMAGE_OPTIONAL_HEADER;

typedef struct _IMAGE_NT_HEADERS {
    ULONG Signature;
    IMAGE_FILE_HEADER FileHeader;
    IMAGE_OPTIONAL_HEADER OptionalHeader;
} IMAGE_NT_HEADERS, *PIMAGE_NT_HEADERS;

//
// Section header format.
//

#define IMAGE_SIZEOF_SHORT_NAME              8

typedef struct _IMAGE_SECTION_HEADER {
    UCHAR   Name[IMAGE_SIZEOF_SHORT_NAME];
    union {
            ULONG   PhysicalAddress;
            ULONG   VirtualSize;
    } Misc;
    ULONG   VirtualAddress;
    ULONG   SizeOfRawData;
    ULONG   PointerToRawData;
    ULONG   PointerToRelocations;
    ULONG   PointerToLinenumbers;
    USHORT  NumberOfRelocations;
    USHORT  NumberOfLinenumbers;
    ULONG   Characteristics;
} IMAGE_SECTION_HEADER, *PIMAGE_SECTION_HEADER;

#include "poppack.h"                        // Return to the default

typedef struct {
    ULONG Function;
    ULONG Toc;
} AIXCALL_FPTR, *PAIXCALL_FPTR;

static const char s_Real0StartPattern[] = "PowerPC";

static LONG HookSignExtBranch(LONG x) {
    return x & 0x2000000 ? (LONG)(x | 0xFC000000) : (LONG)(x);
}

static LONG HookSignExt16(SHORT x) {
    return (LONG)x;
}

// https://github.com/fail0verflow/hbc/blob/a8e5f6c0f7e484c7f7112967eee6eee47b27d9ac/wiipax/stub/sync.c#L29
static void sync_before_exec(const void* p, ULONG len)
{

	ULONG a, b;

	a = (ULONG)p & ~0x1f;
	b = ((ULONG)p + len + 0x1f) & ~0x1f;

	for (; a < b; a += 32)
		asm("dcbst 0,%0 ; sync ; icbi 0,%0" : : "b"(a));

	asm("sync ; isync");
}

// Boyer-Moore Horspool algorithm adapted from http://www-igm.univ-mlv.fr/~lecroq/string/node18.html#SECTION00180
static PBYTE mem_mem(PBYTE startPos, const void* pattern, size_t size, size_t patternSize)
{
    const BYTE* patternc = (const BYTE*)pattern;
    size_t table[256];

    // Preprocessing
    for (ULONG i = 0; i < 256; i++)
        table[i] = patternSize;
    for (size_t i = 0; i < patternSize - 1; i++)
        table[patternc[i]] = patternSize - i - 1;

    // Searching
    size_t j = 0;
    while (j <= size - patternSize)
    {
        BYTE c = startPos[j + patternSize - 1];
        if (patternc[patternSize - 1] == c && memcmp(pattern, startPos + j, patternSize - 1) == 0)
            return startPos + j;
        j += table[c];
    }

    return NULL;
}

// Relocate all code from 0x3010 to 0x2000
// 0x2000 is an unused exception handler (used for PowerPC 601 only)
// Addresses in the 0x3010+ range can get overwritten on real hardware
// (IOS WD sysmodule uses this area, for example)
BOOLEAN HalpFixLowMem(PLOADER_PARAMETER_BLOCK LoaderBlock) {
	// This doesn't need to be done on flipper systems.
	if ((ULONG)RUNTIME_BLOCK[RUNTIME_SYSTEM_TYPE] == ARTX_SYSTEM_FLIPPER) return TRUE;
	
	PLDR_DATA_TABLE_ENTRY Entry = (PLDR_DATA_TABLE_ENTRY)
			LoaderBlock->LoadOrderListHead.Flink;
	ULONG ImageBase = (ULONG)Entry->DllBase;
	
	// Find "PowerPC" string marking beginning of real0.
	PIMAGE_NT_HEADERS NtHeaders = (PIMAGE_NT_HEADERS)(ImageBase + ((PIMAGE_DOS_HEADER)ImageBase)->e_lfanew);
	PIMAGE_FILE_HEADER FileHeader = &NtHeaders->FileHeader;
	USHORT NumberOfSections = FileHeader->NumberOfSections;
	PIMAGE_OPTIONAL_HEADER OptionalHeader = (PIMAGE_OPTIONAL_HEADER)&FileHeader[1];
	PIMAGE_SECTION_HEADER Sections = (PIMAGE_SECTION_HEADER)((size_t)OptionalHeader + FileHeader->SizeOfOptionalHeader);
	
	// real0 is either in .text (NT 3.5x) or in INIT (NT 4.0) section
	PIMAGE_SECTION_HEADER InitSection = (PIMAGE_SECTION_HEADER)(ULONG)NULL;
	// .text is always first section in an nt kernel PE
	PIMAGE_SECTION_HEADER TextSection = &Sections[0];
	if (OptionalHeader->MajorOperatingSystemVersion == 4) {
		for (ULONG i = 0; i < NumberOfSections; i++) {
			if (!strcmp((char*)Sections[i].Name, "INIT")) {
				InitSection = &Sections[i];
				break;
			}
		}
	}
	
	PAIXCALL_FPTR EntryPoint = (PAIXCALL_FPTR)(ImageBase + OptionalHeader->AddressOfEntryPoint);
	ULONG KernelToc = EntryPoint->Toc;
	
	PBYTE Real0 = NULL;
	PVOID InitOrTextAddr = (PVOID)(ImageBase + (InitSection == NULL ? TextSection : InitSection)->VirtualAddress);
	ULONG InitOrTextLength = (InitSection == NULL ? TextSection : InitSection)->Misc.VirtualSize;
	PVOID KernelText = (PVOID)(ImageBase + TextSection->VirtualAddress);
	if (InitSection != NULL) {
		PVOID KernelInitText = (PVOID)(ImageBase + InitSection->VirtualAddress);
		Real0 = mem_mem((PBYTE)KernelInitText, s_Real0StartPattern, InitSection->Misc.VirtualSize, sizeof(s_Real0StartPattern));
	}
	if (Real0 == NULL) {
		Real0 = mem_mem((PBYTE)KernelText, s_Real0StartPattern, TextSection->Misc.VirtualSize, sizeof(s_Real0StartPattern));
	}
	
	if (Real0 == NULL) return FALSE;
	
	ULONG Length = 0;
	PULONG Insn = (PULONG)((ULONG)Real0 + 0x3010);
	while (Insn[0] != 0) {
		if (Insn[0] == KernelToc) {
			Insn++;
			break;
		}
		Insn++;
		Length += sizeof(ULONG);
	}
	
	memcpy(&Real0[0x2000], &Real0[0x3010], Length);
	// Relocate any backwards branches in this range.
	{
		ULONG ExhOffset = 0x2000;
		PPPC_INSTRUCTION ExhInsn = (PPPC_INSTRUCTION)((ULONG)Real0 + 0x2000);
		while (ExhInsn->Long != 0) {
			LONG BranchOffset = 0;
			ULONG BranchAddress = 0, OldBranchAddress = 0;
			if (ExhInsn->Primary_Op == B_OP) {
				BranchOffset = HookSignExtBranch(ExhInsn->Iform_LI << 2);
				BranchAddress = (ULONG)ExhOffset + BranchOffset;
				OldBranchAddress = ExhOffset + (0x3010 - 0x2000) + BranchOffset;
				if (OldBranchAddress < 0x3010) {
					ExhInsn->Iform_LI += ((0x3010 - 0x2000) / sizeof(ULONG));
				}
			}
			else if (ExhInsn->Primary_Op == BC_OP) {
				BranchOffset = HookSignExt16(ExhInsn->Bform_BD << 2);
				BranchAddress = (ULONG)ExhOffset + BranchOffset;
				OldBranchAddress = ExhOffset + (0x3010 - 0x2000) + BranchOffset;
				if (OldBranchAddress < 0x3010) {
					ExhInsn->Iform_LI += ((0x3010 - 0x2000) / sizeof(ULONG));
				}
			}
			ExhInsn++;
			ExhOffset += sizeof(ULONG);
		}
	}
	// Patch all branches in the <0x1000 range to go from +0x3010 to +0x2000
	// This means 0x200, 0x300, 0x400, 0xE00, 0xF00
	static ULONG s_ExhOffsets[] = { 0x200, 0x300, 0x400, 0xE00, 0xF00 };
	for (ULONG ExhIndex = 0; ExhIndex < 5; ExhIndex++) {
		ULONG ExhOffset = s_ExhOffsets[ExhIndex];
		PPPC_INSTRUCTION ExhInsn = (PPPC_INSTRUCTION)((ULONG)Real0 + ExhOffset);
		while (ExhInsn->Long != 0) {
			LONG BranchOffset = 0;
			ULONG BranchAddress = 0;
			if (ExhInsn->Primary_Op == B_OP) {
				BranchOffset = HookSignExtBranch(ExhInsn->Iform_LI << 2);
				BranchAddress = (ULONG)ExhOffset + BranchOffset;
				if (BranchAddress >= 0x3010 && BranchAddress < (0x3010 + Length)) {
					ExhInsn->Iform_LI -= ((0x3010 - 0x2000) / sizeof(ULONG));
				}
			}
			else if (ExhInsn->Primary_Op == BC_OP) {
				BranchOffset = HookSignExt16(ExhInsn->Bform_BD << 2);
				BranchAddress = (ULONG)ExhOffset + BranchOffset;
				if (BranchAddress >= 0x3010 && BranchAddress < (0x3010 + Length)) {
					ExhInsn->Iform_LI -= ((0x3010 - 0x2000) / sizeof(ULONG));
				}
			}
			ExhInsn++;
			ExhOffset += sizeof(ULONG);
		}
		// 0x200 could run straight into 0x300 so allow for that
		if (ExhIndex == 0 && ExhOffset > 0x300) ExhIndex++;
	}
	
	// Copy entire real0.
	memcpy((PVOID)0x80000000, Real0, Length + 0x3010);
	sync_before_exec((PVOID)0x80000000, 0x3800);
	
	return TRUE;
}