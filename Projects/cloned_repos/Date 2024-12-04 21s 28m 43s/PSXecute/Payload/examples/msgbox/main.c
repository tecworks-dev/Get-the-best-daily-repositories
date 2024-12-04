#include <windows.h>

int start()
{
    return MessageBoxA(0, GetCommandLineA(), "PSXecute", 0);
}