/*


  ,ad8888ba,                            88b           d88
 d8"'    `"8b                           888b         d888
d8'        `8b                          88`8b       d8'88
88          88   ,adPPYba,  8b,dPPYba,  88 `8b     d8' 88   ,adPPYba,
88          88  a8"     ""  88P'   "Y8  88  `8b   d8'  88  a8P_____88
Y8,        ,8P  8b          88          88   `8b d8'   88  8PP"""""""
 Y8a.    .a8P   "8a,   ,aa  88          88    `888'    88  "8b,   ,aa
  `"Y8888Y"'     `"Ybbd8"'  88          88     `8'     88   `"Ybbd8"'


Description:
Tool designed to exfiltrate OneDrive Business OCR Data

Requirements:
OneDrive Business. OneDrive for home use does not OCR data (so far)

Usage:
Run this binary from CMD. It accepts 1 argument, the full path to
Microsoft.ListSync.db.

Location:
Microsoft.ListSync.db is located in %LOCALAPPDATA%\Microsoft\OneDrive\ListSync\Business1\settings

Known issues:

1. Code functionality depends on SQLite matching versions. If OneDrive
   decides to upgrade the SQLite version they're using, this code base
   might not work. This code base is using SQLite amalgamation 3.48.
   Likewise, OneDrive business is also using SQLite 3.48

2. If Microsoft.ListSync.db is locked (from being in use) the code may fail.
   It is advised you copy Microsoft.ListSync.db to a different directory prior
   to usage to avoid issues.

Code notes:

1. SQLite functions (and some CRT functions) are defined differently so
   function names matches Microsoft Hungarion syntax. There is no reason for
   me to do this. I'm just really anal about the appearance of my code

2. This code allocates a console when used. This program uses the Windows
   subsystem (wWinMain). This is done to make commandline arguments easier
   to use. In summary, I am once again being really anal about my code.

*/

#include <Windows.h>
#include <stdio.h>
#include <shlwapi.h>
#include "sqlite3.h"

#pragma comment(lib, "Shlwapi.lib")

#define SQLITE3 sqlite3
#define SQLITE3_STATEMENT sqlite3_stmt

#define DatabaseSqlite3Open2 sqlite3_open_v2
#define DatabaseSqlite3Close sqlite3_close
#define DatabaseSqlite3ReturnNull sqlite3_result_null
#define DatabaseSqlite3ReturnInt64 sqlite3_result_int64
#define DatabaseSqlite3GetCharString sqlite3_value_text
#define DatabaseSqlite3GetInt sqlite3_value_int
#define DatabaseSqlite3CreateFunction2 sqlite3_create_function_v2
#define DatabaseSqlite3PrepareQuery2 sqlite3_prepare_v2
#define DatabaseSqlite3Next sqlite3_step
#define DatabaseSqlite3GetColumnText sqlite3_column_text
#define DatabaseSqlite3GetColumnNameString sqlite3_column_name
#define DatabaseSqlite3CloseObject sqlite3_finalize
#define DatabaseSqlite3GetColumnCount sqlite3_column_count

#define FileReopenSecure freopen_s
#define StringToLongLong strtoll
#define StringPrintFormatA snprintf
#define CrtFileOpen fopen_s
#define CrtFileClose fclose
#define CrtFilePrintFormatted fprintf

#define PCBYTE const unsigned char *

SIZE_T WCharStringToCharString(_Inout_ PCHAR Destination, _In_ PWCHAR Source, _In_ SIZE_T MaximumAllowed)
{
	INT Length = (INT)MaximumAllowed;

	while (--Length >= 0)
	{
#pragma warning( push )
#pragma warning( disable : 4244)
		if (!(*Destination++ = *Source++))
			return MaximumAllowed - Length - 1;
#pragma warning( pop ) 
	}

	return MaximumAllowed - Length;
}

INT StringCompareW(_In_ LPCWSTR String1, _In_ LPCWSTR String2)
{
	for (; *String1 == *String2; String1++, String2++)
	{
		if (*String1 == '\0')
			return 0;
	}

	return ((*(LPCWSTR)String1 < *(LPCWSTR)String2) ? -1 : +1);
}

INT StringCompareA(_In_ LPCSTR String1, _In_ LPCSTR String2)
{
	for (; *String1 == *String2; String1++, String2++)
	{
		if (*String1 == '\0')
			return 0;
	}

	return ((*(LPCSTR)String1 < *(LPCSTR)String2) ? -1 : +1);
}

PWCHAR StringCopyW(_Inout_ PWCHAR String1, _In_ LPCWSTR String2)
{
	PWCHAR p = String1;

	while ((*p++ = *String2++) != 0);

	return String1;
}

BOOL MakeDebugConsole(VOID)
{
	FILE* FilePointer = NULL;
	BOOL bFlag = FALSE;

	if (!AllocConsole())
	{
		MessageBoxA(NULL, "Fatal error: Unable to create output console", "Fatal Error", MB_OK);
		return FALSE;
	}

	if (FileReopenSecure(&FilePointer, "CONIN$", "r", stdin) != ERROR_SUCCESS)
		goto EXIT_ROUTINE;

	if (FileReopenSecure(&FilePointer, "CONOUT$", "w", stdout) != ERROR_SUCCESS)
		goto EXIT_ROUTINE;

	if (FileReopenSecure(&FilePointer, "CONOUT$", "w", stderr) != ERROR_SUCCESS)
		goto EXIT_ROUTINE;

	bFlag = TRUE;

EXIT_ROUTINE:

	if (!bFlag)
		MessageBoxA(NULL, "Fatal error: Unable to set STDIO", "Fatal Error", MB_OK);
	else
		printf("[+] Debug output console successfully created\r\n");

	return bFlag;
}

BOOL IsPathValidW(_In_ LPCWSTR FilePath)
{
	HANDLE hFile = INVALID_HANDLE_VALUE;

	hFile = CreateFileW(FilePath, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE)
		return FALSE;

	if (hFile)
		CloseHandle(hFile);

	return TRUE;
}

BOOL ValidateFilePathAndExtension(_In_ LPCWSTR FilePath)
{
	BOOL bFlag = FALSE;
	LPCWSTR Extension = NULL;

	if (!IsPathValidW(FilePath))
	{
		printf("[-] Unable to open file. Make sure the file is copied or not in use.\r\n");
		goto EXIT_ROUTINE;
	}

	Extension = PathFindExtensionW(FilePath);

	if (StringCompareW(Extension, L".db") != ERROR_SUCCESS)
	{
		printf("[-] Unable to verify file. File requires .db extension\r\n");
		goto EXIT_ROUTINE;
	}

	bFlag = TRUE;

EXIT_ROUTINE:

	return bFlag;
}

VOID StringToLongLongCallbackRoutine(sqlite3_context* Context, INT ArgumentCounter, sqlite3_value** ArgumentVector)
{
	LPCSTR String = (LPCSTR)DatabaseSqlite3GetCharString(ArgumentVector[0]);
	INT Base = DatabaseSqlite3GetInt(ArgumentVector[1]);
	PCHAR EndPoint = 0;
	LONGLONG Result = 0;
	BOOL bFlag = FALSE;

	if (String == NULL)
		goto EXIT_ROUTINE;

	Result = StringToLongLong(String, &EndPoint, Base);
	if (EndPoint == String)
		goto EXIT_ROUTINE;

	bFlag = TRUE;

EXIT_ROUTINE:

	if (!bFlag)
		DatabaseSqlite3ReturnNull(Context);
	else
		DatabaseSqlite3ReturnInt64(Context, Result);
}

BOOL DumpOneDriveBusinessOcrData(SQLITE3* DatabaseObject, LPCSTR TableName)
{
	BOOL bFlag = FALSE;
	CHAR Query[MAX_PATH] = { 0 };

	SQLITE3_STATEMENT* StatementObject = NULL;
	INT ReturnValue = 0, Index = 0;

	FILE* FileObject = NULL;

	StringPrintFormatA(Query, sizeof(Query), "SELECT * FROM \"%s\";", TableName);

	printf("[+] Executing query: %s\r\n", Query);
	ReturnValue = DatabaseSqlite3PrepareQuery2(DatabaseObject, Query, -1, &StatementObject, NULL);
	if (ReturnValue != SQLITE_OK)
	{
		printf("[-] Query failed. Rows are not present in SQLite DB or there is a SQLite version mismatch\r\n");
		goto EXIT_ROUTINE;
	}
	else
		printf("[+] Query executed successfully\r\n");


	Index = DatabaseSqlite3GetColumnCount(StatementObject);
	if (Index == 0)
	{
		printf("[-] No data present in table. Skipping...\r\n");
		goto EXIT_ROUTINE;
	}
	else
		printf("[+] Query successfully returned data\r\n");

	printf("[+] Searching for OCR column...\r\n");

	for (INT i = 0; i < Index; i++)
	{
		LPCSTR ColumnString = DatabaseSqlite3GetColumnNameString(StatementObject, i);
		if (ColumnString == NULL)
		{
			printf("[-] Empty column found... continuing\r\n");
			continue;
		}

		if (StringCompareA(ColumnString, "MediaServiceOCR") == 0)
		{
			printf("[+] Column: %s found in %s\r\n", ColumnString, TableName);
			bFlag = TRUE;
		}

	}

	if (!bFlag)
	{
		printf("[-] MediaServiceOCR column not found.\r\n");
		goto EXIT_ROUTINE;
	}
	else
		bFlag = FALSE;

	if (CrtFileOpen(&FileObject, "OcrOutput.csv", "w") != ERROR_SUCCESS)
	{
		printf("[-] Unable to create output CSV\r\n");
		goto EXIT_ROUTINE;
	}


	if (FileObject == NULL)
	{
		printf("[-] Unable to create output CSV\r\n");
		goto EXIT_ROUTINE;
	}
	else
		printf("[+] Output CSV created in current directory\r\n");

	printf("[+] Creating file headers\r\n");
	for (INT i = 0; i < Index; i++)
	{
		LPCSTR ColumnString = DatabaseSqlite3GetColumnNameString(StatementObject, i);
		if (ColumnString == NULL)
			goto EXIT_ROUTINE;

		CrtFilePrintFormatted(FileObject, "\"%s\"", DatabaseSqlite3GetColumnNameString(StatementObject, i));
		if (i < Index - 1)
			CrtFilePrintFormatted(FileObject, ",");
	}

	CrtFilePrintFormatted(FileObject, "\n");

	printf("[+] Writing to CSV\r\n");
	while ((ReturnValue = DatabaseSqlite3Next(StatementObject)) == SQLITE_ROW)
	{
		for (INT i = 0; i < Index; i++)
		{
			PCBYTE TextString = DatabaseSqlite3GetColumnText(StatementObject, i);
			if (TextString == NULL)
				CrtFilePrintFormatted(FileObject, "\"%s\"", "");
			else
				CrtFilePrintFormatted(FileObject, "\"%s\"", TextString);

			if (Index < Index - 1)
				CrtFilePrintFormatted(FileObject, ",");
		}

		CrtFilePrintFormatted(FileObject, ",");
	}

	if (ReturnValue != SQLITE_DONE)
	{
		printf("[-] Unable to walk all rows. Some data may be missing\r\n");
		goto EXIT_ROUTINE;
	}
	else
		printf("[+] Writing complete\r\n");

	bFlag = TRUE;

EXIT_ROUTINE:

	if (StatementObject)
		DatabaseSqlite3CloseObject(StatementObject);

	if (FileObject)
		CrtFileClose(FileObject);

	return bFlag;
}

int WINAPI wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nShowCmd)
{
	BOOL DebugConsole = FALSE;
	BOOL bFlag = FALSE;
	BOOL bIsWalked = FALSE;

	LPWSTR* szArglist = NULL;
	INT Arguments = 0;
	WCHAR DbFilePath[MAX_PATH * sizeof(WCHAR)] = { 0 };
	CHAR SqliteDbFilePath[MAX_PATH * sizeof(WCHAR)] = { 0 };

	SQLITE3* DatabaseObject = NULL;
	INT ReturnValue = 0;

	LPCSTR Query = "SELECT name FROM sqlite_master WHERE name LIKE '%rows';";
	SQLITE3_STATEMENT* StatementObject = NULL;

	if (!MakeDebugConsole())
		goto EXIT_ROUTINE;
	else
		DebugConsole = TRUE;

	szArglist = CommandLineToArgvW(GetCommandLineW(), &Arguments);
	if (szArglist == NULL || Arguments < 2)
	{
		printf("[-] No commandline argument or insufficient arguments.\r\n");
		goto EXIT_ROUTINE;
	}

	if (StringCopyW(DbFilePath, szArglist[1]) == NULL)
	{
		printf("[-] Unable to copy argument buffer. Maybe the path is too long?\r\n");
		goto EXIT_ROUTINE;
	}
	else
		printf("[+] Target found: %ws\r\n", DbFilePath);

	if (!ValidateFilePathAndExtension(DbFilePath))
		goto EXIT_ROUTINE;
	else
		printf("[+] File successfully opened and validated\r\n");

	if (WCharStringToCharString(SqliteDbFilePath, DbFilePath, (MAX_PATH * sizeof(WCHAR))) == 0)
	{
		printf("[-] Unable to transform WCHAR target to CHAR target for SQLite. Maybe the path is too long?\r\n");
		goto EXIT_ROUTINE;
	}
	else
		printf("[+] Target transformed from WCHAR to CHAR for SQLite\r\n");

	ReturnValue = DatabaseSqlite3Open2(SqliteDbFilePath, &DatabaseObject, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, NULL);
	if (ReturnValue != SQLITE_OK)
	{
		printf("[-] Unable to open SQLite database\r\n");
		goto EXIT_ROUTINE;
	}
	else
		printf("[+] Successfully opened SQLite database\r\n");

	ReturnValue = DatabaseSqlite3CreateFunction2(DatabaseObject, "strtoll", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC, NULL, StringToLongLongCallbackRoutine, NULL, NULL, NULL);
	if (ReturnValue != SQLITE_OK)
	{
		printf("[-] Unable to create StringToLongLong callback for target.\r\n");
		goto EXIT_ROUTINE;
	}
	else
		printf("[+] Successfully created StringToLongLong callback for target.\r\n");

	printf("[+] Executing query: %s\r\n", Query);
	ReturnValue = DatabaseSqlite3PrepareQuery2(DatabaseObject, Query, -1, &StatementObject, NULL);
	if (ReturnValue != SQLITE_OK)
	{
		printf("[-] Query failed. Rows are not present in SQLite DB or there is a SQLite version mismatch\r\n");
		goto EXIT_ROUTINE;
	}
	else
		printf("[+] Query executed successfully\r\n");

	printf("[+] Attemping to walk tables...\r\n");
	while ((ReturnValue = DatabaseSqlite3Next(StatementObject)) == SQLITE_ROW)
	{
		PCBYTE Table = DatabaseSqlite3GetColumnText(StatementObject, 0);
		if (Table == NULL)
		{
			printf("[-] Unable to walk tables\r\n");
			goto EXIT_ROUTINE;
		}
		else
			printf("[+] Walking table: %s\r\n", Table);

		bIsWalked = TRUE;

		if (DumpOneDriveBusinessOcrData(DatabaseObject, (const char*)Table))
			break;
	}

	if (!bIsWalked)
		printf("[-] Unabled to walk tables. OCR data may not be present.\r\n");

	bFlag = TRUE;

EXIT_ROUTINE:

	printf("[+] Press ENTER to terminate...\r\n");

#pragma warning( push )
#pragma warning( disable : 6031)
	getchar();
#pragma warning( pop ) 
	if (szArglist)
		LocalFree(szArglist); //lol wtf microsoft

	if (DatabaseObject)
		DatabaseSqlite3Close(DatabaseObject);

	return bFlag ? ERROR_SUCCESS : GetLastError();
}
