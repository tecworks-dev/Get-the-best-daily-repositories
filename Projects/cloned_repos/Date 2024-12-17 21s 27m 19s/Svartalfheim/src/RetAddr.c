/*

    https://www.unknowncheats.me/forum/anti-cheat-bypass/268039-x64-return-address-spoofing-source-explanation.html
    https://github.com/kyleavery/AceLdr/blob/main/src/retaddr.c

*/

#include "Prototypes.h"
#include "Macros.h"
#include "Structs.h"
#include "ntdll.h"

D_SEC(B) PVOID FindGadget(PBYTE pModuleAddr)
{

    for(int i = 0; i < MODULE_SIZE(pModuleAddr); i++)
    {
        if(
            pModuleAddr[i + 0] == 0xFF &&
            pModuleAddr[i + 1] == 0x23
        )
        {
            return (PVOID)(U_PTR(pModuleAddr) + i);
        }
    }

    return NULL;
}

D_SEC(B)
PVOID SpoofRetAddr(PVOID function, PVOID module, PVOID a, PVOID b, PVOID c, PVOID d, PVOID e, PVOID f, PVOID g, PVOID h)
{
    PVOID pTrampoline = FindGadget((PBYTE)module);

    if(pTrampoline != NULL)
    {
        PRM param = { pTrampoline, function, NULL };
        return SpoofStub(a, b, c, d, &param, NULL, e, f, g, h);
    }
    return NULL;

}
