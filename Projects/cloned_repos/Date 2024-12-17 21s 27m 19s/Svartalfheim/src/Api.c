#include "ntdll.h"
#include "Macros.h"
#include "Instance.h"

#define UP      500
#define DOWN    500

/*

    Contains utils functions

*/

D_SEC(B)
DWORD djb2A(PBYTE str)
{
	DWORD dwHash = 0x1337;
	BYTE c;

	while (c = *str++)
	{
		if (c >= 'a' && c <= 'z')
			c -= 'a' - 'A';

		dwHash = ((dwHash << 0x5) + dwHash) + c;
	}
	return dwHash;
}

D_SEC(B)
DWORD djb2W(LPWSTR str)
{
	DWORD dwHash = 0x1337;
	WCHAR c;
	while (c = *str++)
	{
		if (c >= L'a' && c <= L'z')
			c -= L'a' - L'A';
		dwHash = ((dwHash << 0x5) + dwHash) + c;
	}
	return dwHash;
}

D_SEC(B)
PVOID xGetModuleHandle(DWORD dwModuleHash)
{
	PTEB pTeb = (PTEB)__readgsqword(0x30);
	PPEB pPeb = pTeb->ProcessEnvironmentBlock;

	void* firstEntry = pPeb->Ldr->InLoadOrderModuleList.Flink;
	PLIST_ENTRY parser = (PLIST_ENTRY)firstEntry;

	do
	{
		PLDR_DATA_TABLE_ENTRY content = (PLDR_DATA_TABLE_ENTRY)parser;

		if (dwModuleHash == NULL)
		{
			return content->DllBase;
		}

		if (djb2W(content->BaseDllName.Buffer) == dwModuleHash)
		{
			return content->DllBase;
		}

		parser = parser->Flink;
	} while (parser->Flink != firstEntry);

	return NULL;
}

D_SEC(B)
PVOID xGetProcAddress(PVOID pModuleAddr, DWORD dwProcHash)
{
	PIMAGE_DOS_HEADER pDosHeader = (PIMAGE_DOS_HEADER)pModuleAddr;
	if (pDosHeader->e_magic != IMAGE_DOS_SIGNATURE) {
		return NULL;
	}

	PIMAGE_NT_HEADERS pNtHeaders = (PIMAGE_NT_HEADERS)(U_PTR(pModuleAddr) + pDosHeader->e_lfanew);
	if (pNtHeaders->Signature != IMAGE_NT_SIGNATURE) {
		return NULL;
	}

	PIMAGE_EXPORT_DIRECTORY pImgExportDirectory = (PIMAGE_EXPORT_DIRECTORY)(U_PTR(pModuleAddr) + pNtHeaders->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress);
	PDWORD pdwAddressOfFunctions = (PDWORD)((PBYTE)pModuleAddr + pImgExportDirectory->AddressOfFunctions);
	PDWORD pdwAddressOfNames = (PDWORD)((PBYTE)pModuleAddr + pImgExportDirectory->AddressOfNames);
	PWORD pwAddressOfNameOrdinales = (PWORD)((PBYTE)pModuleAddr + pImgExportDirectory->AddressOfNameOrdinals);

	for (int i = 0; i < pImgExportDirectory->NumberOfFunctions; i++)
	{
		PBYTE pczFunctionName = (PBYTE)((PBYTE)pModuleAddr + pdwAddressOfNames[i]);
		if (djb2A(pczFunctionName) == dwProcHash)
		{
			return (PVOID)((PBYTE)pModuleAddr + pdwAddressOfFunctions[pwAddressOfNameOrdinales[i]]);
		}
	}

	return NULL;
}

D_SEC(B)
PVOID GetSyscallInstruction(PVOID searchAddr)
{
	for (int i = 0; i < 500; i++)
	{
		if (
			((PBYTE)searchAddr + i)[0] == 0x0F &&
			((PBYTE)searchAddr + i)[1] == 0x05
			)
		{
			return (PVOID)((PBYTE)searchAddr + i);
		}
	}
	return NULL;
}

D_SEC(B)
BOOL GetSyscall(PVOID pFunctionAddress, PSYS_INFO sysInfo)
{
	if (*((PBYTE)pFunctionAddress) == 0x4c
		&& *((PBYTE)pFunctionAddress + 1) == 0x8b
		&& *((PBYTE)pFunctionAddress + 2) == 0xd1
		&& *((PBYTE)pFunctionAddress + 3) == 0xb8
		&& *((PBYTE)pFunctionAddress + 6) == 0x00
		&& *((PBYTE)pFunctionAddress + 7) == 0x00) {

		BYTE high = *((PBYTE)pFunctionAddress + 5);
		BYTE low = *((PBYTE)pFunctionAddress + 4);
		sysInfo->syscall = (high << 8) | low;
		sysInfo->pAddress = GetSyscallInstruction(pFunctionAddress);
		return TRUE;
	}
	else {
		for (WORD idx = 1; idx <= 500; idx++) {
			if (*((PBYTE)pFunctionAddress + idx * DOWN) == 0x4c
				&& *((PBYTE)pFunctionAddress + 1 + idx * DOWN) == 0x8b
				&& *((PBYTE)pFunctionAddress + 2 + idx * DOWN) == 0xd1
				&& *((PBYTE)pFunctionAddress + 3 + idx * DOWN) == 0xb8
				&& *((PBYTE)pFunctionAddress + 6 + idx * DOWN) == 0x00
				&& *((PBYTE)pFunctionAddress + 7 + idx * DOWN) == 0x00) {
				BYTE high = *((PBYTE)pFunctionAddress + 5 + idx * DOWN);
				BYTE low = *((PBYTE)pFunctionAddress + 4 + idx * DOWN);
				sysInfo->syscall = (high << 8) | low - idx;
				sysInfo->pAddress = GetSyscallInstruction((PBYTE)pFunctionAddress + idx * DOWN);

				return TRUE;
			}
			if (*((PBYTE)pFunctionAddress + idx * UP) == 0x4c
				&& *((PBYTE)pFunctionAddress + 1 + idx * UP) == 0x8b
				&& *((PBYTE)pFunctionAddress + 2 + idx * UP) == 0xd1
				&& *((PBYTE)pFunctionAddress + 3 + idx * UP) == 0xb8
				&& *((PBYTE)pFunctionAddress + 6 + idx * UP) == 0x00
				&& *((PBYTE)pFunctionAddress + 7 + idx * UP) == 0x00) {
				BYTE high = *((PBYTE)pFunctionAddress + 5 + idx * UP);
				BYTE low = *((PBYTE)pFunctionAddress + 4 + idx * UP);
				sysInfo->syscall = (high << 8) | low + idx;
				sysInfo->pAddress = GetSyscallInstruction((PBYTE)pFunctionAddress + idx * UP);

				return TRUE;
			}
		}
		return FALSE;
	}
	
	return FALSE;
}

D_SEC(B)
VOID xMemcpy(PBYTE dst, PBYTE src, DWORD size)
{
	while (size--)
		dst[size] = src[size];
}

D_SEC(B)
VOID xMemset(PBYTE dst, BYTE c, DWORD size)
{
	while (size--)
		dst[size] = c;
}