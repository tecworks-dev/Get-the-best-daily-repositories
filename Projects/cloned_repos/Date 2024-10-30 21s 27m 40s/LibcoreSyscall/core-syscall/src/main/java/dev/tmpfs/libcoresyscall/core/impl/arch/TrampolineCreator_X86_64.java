package dev.tmpfs.libcoresyscall.core.impl.arch;

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import dev.tmpfs.libcoresyscall.core.impl.NativeBridge;
import dev.tmpfs.libcoresyscall.core.impl.ReflectHelper;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.BaseTrampolineCreator;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.ISyscallNumberTable;

public class TrampolineCreator_X86_64 extends BaseTrampolineCreator implements ISyscallNumberTable {

    private TrampolineCreator_X86_64() {
    }

    public static final TrampolineCreator_X86_64 INSTANCE = new TrampolineCreator_X86_64();

    @Override
    public byte[] getPaddingInstruction() {
        // int3
        return new byte[]{(byte) 0xcc};
    }

    @Override
    public Map<Method, byte[]> getNativeMethods() {
        HashMap<Method, byte[]> result = new HashMap<>();
        try {
            //0000000000000000 <NativeBridge_nativeClearCache>:
            //       0: c3                            retq
            //       1: 66 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00  nopw    %cs:(%rax,%rax)
            result.put(
                    NativeBridge.class.getMethod("nativeClearCache", long.class, long.class),
                    new byte[]{
                            // retq
                            (byte) 0xc3,
                            // nopw    %cs:(%rax,%rax)
                            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, (byte) 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
                    }
            );
            //0000000000000010 <NativeBridge_nativeCallPointerFunction0>:
            //      10: 31 c0                         xorl    %eax, %eax
            //      12: ff e2                         jmpq    *%rdx
            //      14: 66 66 66 2e 0f 1f 84 00 00 00 00 00   nopw    %cs:(%rax,%rax)
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction0", long.class),
                    new byte[]{
                            // xorl    %eax, %eax
                            0x31, (byte) 0xc0,
                            // jmpq    *%rdx
                            (byte) 0xff, (byte) 0xe2,
                            // nopw    %cs:(%rax,%rax)
                            0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, (byte) 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
                    }
            );
            //00000000000018c0 <NativeBridge_nativeCallPointerFunction1>:
            //    18c0: 48 89 cf                      movq    %rcx, %rdi
            //    18c3: ff e2                         jmpq    *%rdx
            //    18c5: 66 66 2e 0f 1f 84 00 00 00 00 00      nopw    %cs:(%rax,%rax)
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction1", long.class, long.class),
                    new byte[]{
                            // movq    %rcx, %rdi
                            0x48, (byte) 0x89, (byte) 0xcf,
                            // jmpq    *%rdx
                            (byte) 0xff, (byte) 0xe2,
                            // nopw    %cs:(%rax,%rax)
                            0x66, 0x66, 0x2e, 0x0f, 0x1f, (byte) 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
                    }
            );
            //0000000000000030 <NativeBridge_nativeCallPointerFunction2>:
            //      30: 4c 89 c6                      movq    %r8, %rsi
            //      33: 48 89 cf                      movq    %rcx, %rdi
            //      36: ff e2                         jmpq    *%rdx
            //      38: 0f 1f 84 00 00 00 00 00       nopl    (%rax,%rax)
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction2", long.class, long.class, long.class),
                    new byte[]{
                            // movq    %r8, %rsi
                            0x4c, (byte) 0x89, (byte) 0xc6,
                            // movq    %rcx, %rdi
                            0x48, (byte) 0x89, (byte) 0xcf,
                            // jmpq    *%rdx
                            (byte) 0xff, (byte) 0xe2,
                            // nopl    (%rax,%rax)
                            0x0f, 0x1f, (byte) 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
                    }
            );
            //0000000000000040 <NativeBridge_nativeCallPointerFunction3>:
            //      40: 4c 89 c6                      movq    %r8, %rsi
            //      43: 48 89 d0                      movq    %rdx, %rax
            //      46: 48 89 cf                      movq    %rcx, %rdi
            //      49: 4c 89 ca                      movq    %r9, %rdx
            //      4c: ff e0                         jmpq    *%rax
            //      4e: 66 90                         nop
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction3", long.class, long.class, long.class, long.class),
                    new byte[]{
                            // movq    %r8, %rsi
                            0x4c, (byte) 0x89, (byte) 0xc6,
                            // movq    %rdx, %rax
                            0x48, (byte) 0x89, (byte) 0xd0,
                            // movq    %rcx, %rdi
                            0x48, (byte) 0x89, (byte) 0xcf,
                            // movq    %r9, %rdx
                            0x4c, (byte) 0x89, (byte) 0xca,
                            // jmpq    *%rax
                            (byte) 0xff, (byte) 0xe0,
                            // nop
                            0x66, (byte) 0x90
                    }
            );
            //0000000000000050 <NativeBridge_nativeCallPointerFunction4>:
            //      50: 4c 89 c6                      movq    %r8, %rsi
            //      53: 48 89 d0                      movq    %rdx, %rax
            //      56: 4c 8b 44 24 08                movq    0x8(%rsp), %r8
            //      5b: 48 89 cf                      movq    %rcx, %rdi
            //      5e: 4c 89 ca                      movq    %r9, %rdx
            //      61: 4c 89 c1                      movq    %r8, %rcx
            //      64: ff e0                         jmpq    *%rax
            //      66: 66 2e 0f 1f 84 00 00 00 00 00 nopw    %cs:(%rax,%rax)
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction4", long.class, long.class, long.class, long.class, long.class),
                    new byte[]{
                            // movq    %r8, %rsi
                            0x4c, (byte) 0x89, (byte) 0xc6,
                            // movq    %rdx, %rax
                            0x48, (byte) 0x89, (byte) 0xd0,
                            // movq    0x8(%rsp), %r8
                            0x4c, (byte) 0x8b, 0x44, 0x24, 0x08,
                            // movq    %rcx, %rdi
                            0x48, (byte) 0x89, (byte) 0xcf,
                            // movq    %r9, %rdx
                            0x4c, (byte) 0x89, (byte) 0xca,
                            // movq    %r8, %rcx
                            0x4c, (byte) 0x89, (byte) 0xc1,
                            // jmpq    *%rax
                            (byte) 0xff, (byte) 0xe0,
                            // nopw    %cs:(%rax,%rax)
                            0x66, 0x2e, 0x0f, 0x1f, (byte) 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
                    }
            );
            //0000000000001690 <NativeBridge_nativeGetJavaVM>:
            //    1690: 50                            pushq   %rax
            //    1691: 48 c7 04 24 00 00 00 00       movq    $0x0, (%rsp)
            //    1699: 48 8b 07                      movq    (%rdi), %rax
            //    169c: 48 89 e6                      movq    %rsp, %rsi
            //    169f: ff 90 d8 06 00 00             callq   *0x6d8(%rax)
            //    16a5: 85 c0                         testl   %eax, %eax
            //    16a7: 75 06                         jne     0x16af <NativeBridge_nativeGetJavaVM+0x1f>
            //    16a9: 48 8b 04 24                   movq    (%rsp), %rax
            //    16ad: 59                            popq    %rcx
            //    16ae: c3                            retq
            //    16af: 31 c0                         xorl    %eax, %eax
            //    16b1: 59                            popq    %rcx
            //    16b2: c3                            retq
            //    16b3: 66 66 66 66 2e 0f 1f 84 00 00 00 00 00        nopw    %cs:(%rax,%rax)
            result.put(
                    NativeBridge.class.getMethod("nativeGetJavaVM"),
                    new byte[]{
                            // pushq   %rax
                            0x50,
                            // movq    $0x0, (%rsp)
                            0x48, (byte) 0xc7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00,
                            // movq    (%rdi), %rax
                            0x48, (byte) 0x8b, 0x07,
                            // movq    %rsp, %rsi
                            0x48, (byte) 0x89, (byte) 0xe6,
                            // callq   *0x6d8(%rax)
                            (byte) 0xff, (byte) 0x90, (byte) 0xd8, 0x06, 0x00, 0x00,
                            // testl   %eax, %eax
                            (byte) 0x85, (byte) 0xc0,
                            // jne     0x16af <NativeBridge_nativeGetJavaVM+0x1f>
                            (byte) 0x75, 0x06,
                            // movq    (%rsp), %rax
                            0x48, (byte) 0x8b, 0x04, 0x24,
                            // popq    %rcx
                            0x59,
                            // retq
                            (byte) 0xc3,
                            // xorl    %eax, %eax
                            0x31, (byte) 0xc0,
                            // popq    %rcx
                            0x59,
                            // retq
                            (byte) 0xc3,
                            // nopw    %cs:(%rax,%rax)
                            0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, (byte) 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
                    }
            );
            //0000000000000070 <NativeBridge_nativeSyscall>:
            //      70: 41 57                         pushq   %r15
            //      72: 41 56                         pushq   %r14
            //      74: 53                            pushq   %rbx
            //      75: 4c 89 c6                      movq    %r8, %rsi
            //      78: 48 8b 5c 24 20                movq    0x20(%rsp), %rbx
            //      7d: 4c 8b 74 24 28                movq    0x28(%rsp), %r14
            //      82: 4c 8b 7c 24 30                movq    0x30(%rsp), %r15
            //      87: 48 63 c2                      movslq  %edx, %rax
            //      8a: 48 89 cf                      movq    %rcx, %rdi
            //      8d: 4c 89 ca                      movq    %r9, %rdx
            //      90: 49 89 da                      movq    %rbx, %r10
            //      93: 4d 89 f0                      movq    %r14, %r8
            //      96: 4d 89 f9                      movq    %r15, %r9
            //      99: 0f 05                         syscall
            //      9b: 5b                            popq    %rbx
            //      9c: 41 5e                         popq    %r14
            //      9e: 41 5f                         popq    %r15
            //      a0: c3                            retq
            //      a1: 66 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00  nopw    %cs:(%rax,%rax)
            result.put(
                    NativeBridge.class.getMethod("nativeSyscall", int.class, long.class, long.class, long.class, long.class, long.class, long.class),
                    new byte[]{
                            // pushq   %r15
                            0x41, 0x57,
                            // pushq   %r14
                            0x41, 0x56,
                            // pushq   %rbx
                            0x53,
                            // movq    %r8, %rsi
                            0x4c, (byte) 0x89, (byte) 0xc6,
                            // movq    0x20(%rsp), %rbx
                            0x48, (byte) 0x8b, 0x5c, 0x24, 0x20,
                            // movq    0x28(%rsp), %r14
                            0x4c, (byte) 0x8b, 0x74, 0x24, 0x28,
                            // movq    0x30(%rsp), %r15
                            0x4c, (byte) 0x8b, 0x7c, 0x24, 0x30,
                            // movslq  %edx, %rax
                            0x48, 0x63, (byte) 0xc2,
                            // movq    %rcx, %rdi
                            0x48, (byte) 0x89, (byte) 0xcf,
                            // movq    %r9, %rdx
                            0x4c, (byte) 0x89, (byte) 0xca,
                            // movq    %rbx, %r10
                            0x49, (byte) 0x89, (byte) 0xda,
                            // movq    %r14, %r8
                            0x4d, (byte) 0x89, (byte) 0xf0,
                            // movq    %r15, %r9
                            0x4d, (byte) 0x89, (byte) 0xf9,
                            // syscall
                            0x0f, 0x05,
                            // popq    %rbx
                            0x5b,
                            // popq    %r14
                            0x41, 0x5e,
                            // popq    %r15
                            0x41, 0x5f,
                            // retq
                            (byte) 0xc3,
                            // nopw    %cs:(%rax,%rax)
                            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, (byte) 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
                    }
            );
        } catch (NoSuchMethodException e) {
            ReflectHelper.unsafeThrow(e);
        }
        return Collections.unmodifiableMap(result);
    }

    @Override
    public int __NR_mprotect() {
        // mprotect x86_64: 10
        return 10;
    }

    @Override
    public int __NR_memfd_create() {
        // memfd_create x86_64: 319
        return 319;
    }
}
