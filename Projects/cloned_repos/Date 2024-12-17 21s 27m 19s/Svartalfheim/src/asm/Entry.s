[BITS 64]

extern PreMain

global EntryPoint

[SECTION .text$A]

    EntryPoint:
        push rsi                            
        mov rsi, rsp                        
        and   rsp, 0FFFFFFFFFFFFFFF0h       
        sub   rsp, 0x20                     
        call  PreMain                   
        mov   rsp, rsi                      
        pop   rsi                           
        ret               
