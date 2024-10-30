package dev.tmpfs.libcoresyscall.core.impl.arch;

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import dev.tmpfs.libcoresyscall.core.impl.NativeBridge;
import dev.tmpfs.libcoresyscall.core.impl.ReflectHelper;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.BaseTrampolineCreator;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.ISyscallNumberTable;

public class TrampolineCreator_Mipsel extends BaseTrampolineCreator implements ISyscallNumberTable {

    private TrampolineCreator_Mipsel() {
    }

    public static final TrampolineCreator_Mipsel INSTANCE = new TrampolineCreator_Mipsel();

    private static byte[] instructionsToBytes(int... inst) {
        byte[] result = new byte[inst.length * 4];
        for (int i = 0; i < inst.length; i++) {
            result[i * 4] = (byte) (inst[i] & 0xff);
            result[i * 4 + 1] = (byte) ((inst[i] >> 8) & 0xff);
            result[i * 4 + 2] = (byte) ((inst[i] >> 16) & 0xff);
            result[i * 4 + 3] = (byte) ((inst[i] >> 24) & 0xff);
        }
        return result;
    }

    @Override
    public byte[] getPaddingInstruction() {
        //000004c8 <NativeBridge_breakpoint>:
        // 4c8:   0000000d        break
        return instructionsToBytes(0x0000000d);
    }

    @Override
    public Map<Method, byte[]> getNativeMethods() {
        HashMap<Method, byte[]> result = new HashMap<>();
        try {
            //000004d0 <NativeBridge_nativeClearCache>:
            // 4d0:   7c02083b        0x7c02083b
            // 4d4:   14400003        bnez    v0,4e4 <NativeBridge_nativeClearCache+0x14>
            // 4d8:   00000000        nop
            // 4dc:   03e00008        jr      ra
            // 4e0:   00000000        nop
            // 4e4:   0000000d        break
            // XXX: Is this correct?
            result.put(
                    NativeBridge.class.getMethod("nativeClearCache", long.class, long.class),
                    instructionsToBytes(
                            0x7c02083b,
                            0x14400003,
                            0x00000000,
                            0x03e00008,
                            0x00000000,
                            0x0000000d
                    )
            );
            //000004e8 <NativeBridge_nativeCallPointerFunction0>:
            // 4e8:   27bdffe8        addiu   sp,sp,-24
            // 4ec:   afbf0014        sw      ra,20(sp)
            // 4f0:   00c0c825        move    t9,a2
            // 4f4:   0320f809        jalr    t9
            // 4f8:   00000000        nop
            // 4fc:   8fbf0014        lw      ra,20(sp)
            // 500:   24030000        li      v1,0
            // 504:   03e00008        jr      ra
            // 508:   27bd0018        addiu   sp,sp,24
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction0", long.class),
                    instructionsToBytes(
                            0x27bdffe8,
                            0xafbf0014,
                            0x00c0c825,
                            0x0320f809,
                            0x00000000,
                            0x8fbf0014,
                            0x24030000,
                            0x03e00008,
                            0x27bd0018
                    )
            );
            //0000050c <NativeBridge_nativeCallPointerFunction1>:
            // 50c:   27bdffe8        addiu   sp,sp,-24
            // 510:   afbf0014        sw      ra,20(sp)
            // 514:   8fa40028        lw      a0,40(sp)
            // 518:   00c0c825        move    t9,a2
            // 51c:   0320f809        jalr    t9
            // 520:   00000000        nop
            // 524:   8fbf0014        lw      ra,20(sp)
            // 528:   24030000        li      v1,0
            // 52c:   03e00008        jr      ra
            // 530:   27bd0018        addiu   sp,sp,24
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction1", long.class, long.class),
                    instructionsToBytes(
                            0x27bdffe8,
                            0xafbf0014,
                            0x8fa40028,
                            0x00c0c825,
                            0x0320f809,
                            0x00000000,
                            0x8fbf0014,
                            0x24030000,
                            0x03e00008,
                            0x27bd0018
                    )
            );
            //00000534 <NativeBridge_nativeCallPointerFunction2>:
            // 534:   27bdffe8        addiu   sp,sp,-24
            // 538:   afbf0014        sw      ra,20(sp)
            // 53c:   8fa50030        lw      a1,48(sp)
            // 540:   8fa40028        lw      a0,40(sp)
            // 544:   00c0c825        move    t9,a2
            // 548:   0320f809        jalr    t9
            // 54c:   00000000        nop
            // 550:   8fbf0014        lw      ra,20(sp)
            // 554:   24030000        li      v1,0
            // 558:   03e00008        jr      ra
            // 55c:   27bd0018        addiu   sp,sp,24
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction2", long.class, long.class, long.class),
                    instructionsToBytes(
                            0x27bdffe8,
                            0xafbf0014,
                            0x8fa50030,
                            0x8fa40028,
                            0x00c0c825,
                            0x0320f809,
                            0x00000000,
                            0x8fbf0014,
                            0x24030000,
                            0x03e00008,
                            0x27bd0018
                    )
            );
            //00000560 <NativeBridge_nativeCallPointerFunction3>:
            // 560:   27bdffe8        addiu   sp,sp,-24
            // 564:   afbf0014        sw      ra,20(sp)
            // 568:   00c00825        move    at,a2
            // 56c:   8fa60038        lw      a2,56(sp)
            // 570:   8fa50030        lw      a1,48(sp)
            // 574:   8fa40028        lw      a0,40(sp)
            // 578:   0020c825        move    t9,at
            // 57c:   0320f809        jalr    t9
            // 580:   00000000        nop
            // 584:   8fbf0014        lw      ra,20(sp)
            // 588:   24030000        li      v1,0
            // 58c:   03e00008        jr      ra
            // 590:   27bd0018        addiu   sp,sp,24
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction3", long.class, long.class, long.class, long.class),
                    instructionsToBytes(
                            0x27bdffe8,
                            0xafbf0014,
                            0x00c00825,
                            0x8fa60038,
                            0x8fa50030,
                            0x8fa40028,
                            0x0020c825,
                            0x0320f809,
                            0x00000000,
                            0x8fbf0014,
                            0x24030000,
                            0x03e00008,
                            0x27bd0018
                    )
            );
            //00000594 <NativeBridge_nativeCallPointerFunction4>:
            // 594:   27bdffe8        addiu   sp,sp,-24
            // 598:   afbf0014        sw      ra,20(sp)
            // 59c:   00c00825        move    at,a2
            // 5a0:   8fa70040        lw      a3,64(sp)
            // 5a4:   8fa60038        lw      a2,56(sp)
            // 5a8:   8fa50030        lw      a1,48(sp)
            // 5ac:   8fa40028        lw      a0,40(sp)
            // 5b0:   0020c825        move    t9,at
            // 5b4:   0320f809        jalr    t9
            // 5b8:   00000000        nop
            // 5bc:   8fbf0014        lw      ra,20(sp)
            // 5c0:   24030000        li      v1,0
            // 5c4:   03e00008        jr      ra
            // 5c8:   27bd0018        addiu   sp,sp,24
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction4", long.class, long.class, long.class, long.class, long.class),
                    instructionsToBytes(
                            0x27bdffe8,
                            0xafbf0014,
                            0x00c00825,
                            0x8fa70040,
                            0x8fa60038,
                            0x8fa50030,
                            0x8fa40028,
                            0x0020c825,
                            0x0320f809,
                            0x00000000,
                            0x8fbf0014,
                            0x24030000,
                            0x03e00008,
                            0x27bd0018
                    )
            );
            //000005cc <NativeBridge_nativeGetJavaVM>:
            // 5cc:   27bdffe8        addiu   sp,sp,-24
            // 5d0:   afbf0014        sw      ra,20(sp)
            // 5d4:   afa00010        sw      zero,16(sp)
            // 5d8:   8c810000        lw      at,0(a0)
            // 5dc:   8c39036c        lw      t9,876(at)
            // 5e0:   0320f809        jalr    t9
            // 5e4:   27a50010        addiu   a1,sp,16
            // 5e8:   8fa10010        lw      at,16(sp)
            // 5ec:   8fbf0014        lw      ra,20(sp)
            // 5f0:   24030000        li      v1,0
            // 5f4:   0002080b        movn    at,zero,v0
            // 5f8:   00201025        move    v0,at
            // 5fc:   03e00008        jr      ra
            // 600:   27bd0018        addiu   sp,sp,24
            result.put(
                    NativeBridge.class.getMethod("nativeGetJavaVM"),
                    instructionsToBytes(
                            0x27bdffe8,
                            0xafbf0014,
                            0xafa00010,
                            0x8c810000,
                            0x8c39036c,
                            0x0320f809,
                            0x27a50010,
                            0x8fa10010,
                            0x8fbf0014,
                            0x24030000,
                            0x0002080b,
                            0x00201025,
                            0x03e00008,
                            0x27bd0018
                    )
            );
            //00000480 <NativeBridge_nativeSyscall>:
            // 480:   00c00825        move    at,a2
            // 484:   8fa30038        lw      v1,56(sp)
            // 488:   8fbc0030        lw      gp,48(sp)
            // 48c:   8fa70028        lw      a3,40(sp)
            // 490:   8fa60020        lw      a2,32(sp)
            // 494:   8fa50018        lw      a1,24(sp)
            // 498:   8fa40010        lw      a0,16(sp)
            // 49c:   00201025        move    v0,at
            // 4a0:   27bdffe0        addiu   sp,sp,-32
            // 4a4:   afbc0010        sw      gp,16(sp)
            // 4a8:   afa30014        sw      v1,20(sp)
            // 4ac:   0000000c        syscall
            // 4b0:   27bd0020        addiu   sp,sp,32
            // 4b4:   24030000        li      v1,0
            // 4b8:   00020823        negu    at,v0
            // 4bc:   0047080a        movz    at,v0,a3
            // 4c0:   03e00008        jr      ra
            // 4c4:   00201025        move    v0,at
            result.put(
                    NativeBridge.class.getMethod("nativeSyscall", int.class, long.class, long.class, long.class, long.class, long.class, long.class),
                    instructionsToBytes(
                            0x00c00825,
                            0x8fa30038,
                            0x8fbc0030,
                            0x8fa70028,
                            0x8fa60020,
                            0x8fa50018,
                            0x8fa40010,
                            0x00201025,
                            0x27bdffe0,
                            0xafbc0010,
                            0xafa30014,
                            0x0000000c,
                            0x27bd0020,
                            0x24030000,
                            0x00020823,
                            0x0047080a,
                            0x03e00008,
                            0x00201025
                    )
            );
        } catch (NoSuchMethodException e) {
            ReflectHelper.unsafeThrow(e);
        }
        return Collections.unmodifiableMap(result);
    }

    @Override
    public int __NR_mprotect() {
        // mprotect mipsel o32 4000+125
        return 4125;
    }

    @Override
    public int __NR_memfd_create() {
        // memfd_create mipsel o32 4000+354
        return 4354;
    }
}
