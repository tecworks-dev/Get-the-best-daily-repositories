#define D_API( x )  __typeof__( x ) * x
#define D_SEC( x )  __attribute__( ( section( ".text$" #x "" ) ) )
#define MODULE_SIZE(x)      ((PIMAGE_NT_HEADERS)((UINT_PTR)x + ((PIMAGE_DOS_HEADER)x)->e_lfanew))->OptionalHeader.SizeOfImage
#define DBREAK              __debugbreak()

// Cast macro
#define U_PTR(x)   (UINT_PTR)x

// Copy/Pasta here https://github.com/kyleavery/AceLdr/blob/main/src/include.h
#define SPOOF_X( function, module )                            SpoofRetAddr( function, module, NULL, NULL, NULL, NULL, NULL, NULL, NULL )
#define SPOOF_A( function, module, a )                          SpoofRetAddr( function, module, a, NULL, NULL, NULL, NULL, NULL, NULL, NULL )
#define SPOOF_B( function, module, a, b )                       SpoofRetAddr( function, module, a, b, NULL, NULL, NULL, NULL, NULL, NULL )
#define SPOOF_C( function, module, a, b, c )                    SpoofRetAddr( function, module, a, b, c, NULL, NULL, NULL, NULL, NULL )
#define SPOOF_D( function, module, a, b, c, d )                 SpoofRetAddr( function, module, a, b, c, d, NULL, NULL, NULL, NULL )
#define SPOOF_E( function, module, a, b, c, d, e )              SpoofRetAddr( function, module, a, b, c, d, e, NULL, NULL, NULL )
#define SPOOF_F( function, module, a, b, c, d, e, f )           SpoofRetAddr( function, module, a, b, c, d, e, f, NULL, NULL )
#define SPOOF_G( function, module, a, b, c, d, e, f, g )        SpoofRetAddr( function, module, a, b, c, d, e, f, g, NULL )
#define SPOOF_H( function, module, a, b, c, d, e, f, g, h )     SpoofRetAddr( function, module, a, b, c, d, e, f, g, h )
#define SETUP_ARGS(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, ...) arg11
#define SPOOF_MACRO_CHOOSER(...) SETUP_ARGS(__VA_ARGS__, SPOOF_H, SPOOF_G, SPOOF_F, SPOOF_E, SPOOF_D, SPOOF_C, SPOOF_B, SPOOF_A, SPOOF_X, )
#define SPOOF(...) SPOOF_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

#define NTCALL(x, ...) do {                     \
    RegBackup(&regContent);                    \
    SysPrepare(Inst->Sys.x.syscall,            \
               Inst->Sys.x.pAddress);          \
    SysCall(__VA_ARGS__);                      \
    RegRestore(&regContent);                   \
} while(0)

typedef void (WINAPI* EXEC_MEM)();
