#include <string>
#include <unordered_map>

std::unordered_map<std::string, std::string> WindowsAPIMapping = {
    // User32 DLL
    {"MessageBoxA", "user32.dll"},

    // Kernel32 DLL
    {"CreateFileA", "kernel32.dll"},
    {"ReadFile", "kernel32.dll"},
    {"WriteFile", "kernel32.dll"},
    {"CloseHandle", "kernel32.dll"},
    {"GetLastError", "kernel32.dll"},
    {"SetLastError", "kernel32.dll"},
    {"LoadLibraryA", "kernel32.dll"},
    {"GetProcAddress", "kernel32.dll"},
    {"FreeLibrary", "kernel32.dll"},
    {"VirtualAlloc", "kernel32.dll"},
    {"VirtualFree", "kernel32.dll"},
    {"GetProcessHeap", "kernel32.dll"},
    {"HeapAlloc", "kernel32.dll"},
    {"GetCommandLineA", "kernel32.dll"},

    // Secur32 DLL
    {"GetUserNameExA", "secur32.dll"},

    // Advapi32 DLL
    {"RegOpenKeyExA", "advapi32.dll"},
    {"RegQueryValueExA", "advapi32.dll"},
    {"RegCloseKey", "advapi32.dll"},
    {"RegCreateKeyExA", "advapi32.dll"},

    // Ws2_32 DLL (Winsock)
    {"socket", "ws2_32.dll"},
    {"bind", "ws2_32.dll"},
    {"listen", "ws2_32.dll"},
    {"accept", "ws2_32.dll"},
    {"connect", "ws2_32.dll"},
    {"send", "ws2_32.dll"},
    {"recv", "ws2_32.dll"},
    {"closesocket", "ws2_32.dll"},

    // Ole32 DLL
    {"CoInitializeEx", "ole32.dll"},
    {"CoUninitialize", "ole32.dll"},
    {"CoCreateInstance", "ole32.dll"},

    // C Runtime Library
    {"malloc", "msvcrt.dll"},
    {"free", "msvcrt.dll"},
    {"printf", "msvcrt.dll"},
    {"scanf", "msvcrt.dll"},
    {"fopen", "msvcrt.dll"},
    {"fclose", "msvcrt.dll"},
    {"fread", "msvcrt.dll"},
    {"fwrite", "msvcrt.dll"}};