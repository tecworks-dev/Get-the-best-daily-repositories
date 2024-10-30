package dev.tmpfs.libcoresyscall.core.impl.arch;

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import dev.tmpfs.libcoresyscall.core.impl.NativeBridge;
import dev.tmpfs.libcoresyscall.core.impl.ReflectHelper;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.BaseTrampolineCreator;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.ISyscallNumberTable;

public class TrampolineCreator_Riscv64 extends BaseTrampolineCreator implements ISyscallNumberTable {

    private TrampolineCreator_Riscv64() {
    }

    public static final TrampolineCreator_Riscv64 INSTANCE = new TrampolineCreator_Riscv64();

    @Override
    public byte[] getPaddingInstruction() {
        //0000000000001696 <NativeBridge_breakpoint>:
        //    1696: 02 90         ebreak
        return new byte[]{0x02, (byte) 0x90};
    }

    @Override
    public Map<Method, byte[]> getNativeMethods() {
        HashMap<Method, byte[]> result = new HashMap<>();
        try {
            //00000000000015dc <NativeBridge_nativeClearCache>:
            //    15dc: 32 85         mv      a0, a2
            //    15de: b3 85 c6 00   add     a1, a3, a2
            //    15e2: 93 08 30 10   li      a7, 0x103
            //    15e6: 01 46         li      a2, 0x0
            //    15e8: 73 00 00 00   ecall
            //    15ec: 11 e1         bnez    a0, 0x15f0 <NativeBridge_nativeClearCache+0x14>
            //    15ee: 82 80         ret
            //    15f0: 00 00         unimp
            result.put(
                    NativeBridge.class.getMethod("nativeClearCache", long.class, long.class),
                    new byte[]{
                            0x32, (byte) 0x85,
                            (byte) 0xb3, (byte) 0x85, (byte) 0xc6, 0x00,
                            (byte) 0x93, 0x08, 0x30, 0x10,
                            0x01, 0x46,
                            0x73, 0x00, 0x00, 0x00,
                            0x11, (byte) 0xe1,
                            (byte) 0x82, (byte) 0x80,
                            0x00, 0x00
                    }
            );
            //00000000000015f2 <NativeBridge_nativeCallPointerFunction0>:
            //    15f2: 02 86         jr      a2
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction0", long.class),
                    new byte[]{0x02, (byte) 0x86}
            );
            //00000000000015f4 <NativeBridge_nativeCallPointerFunction1>:
            //    15f4: 36 85         mv      a0, a3
            //    15f6: 02 86         jr      a2
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction1", long.class, long.class),
                    new byte[]{(byte) 0x36, (byte) 0x85, 0x02, (byte) 0x86}
            );
            //00000000000015f8 <NativeBridge_nativeCallPointerFunction2>:
            //    15f8: ba 85         mv      a1, a4
            //    15fa: 36 85         mv      a0, a3
            //    15fc: 02 86         jr      a2
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction2", long.class, long.class, long.class),
                    new byte[]{(byte) 0xba, (byte) 0x85, (byte) 0x36, (byte) 0x85, 0x02, (byte) 0x86}
            );
            //00000000000015fe <NativeBridge_nativeCallPointerFunction3>:
            //    15fe: ba 85         mv      a1, a4
            //    1600: 36 85         mv      a0, a3
            //    1602: 32 87         mv      a4, a2
            //    1604: 3e 86         mv      a2, a5
            //    1606: 02 87         jr      a4
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction3", long.class, long.class, long.class, long.class),
                    new byte[]{
                            (byte) 0xba, (byte) 0x85,
                            (byte) 0x36, (byte) 0x85,
                            0x32, (byte) 0x87,
                            0x3e, (byte) 0x86,
                            0x02, (byte) 0x87
                    }
            );
            //0000000000001608 <NativeBridge_nativeCallPointerFunction4>:
            //    1608: ba 85         mv      a1, a4
            //    160a: 36 85         mv      a0, a3
            //    160c: 32 87         mv      a4, a2
            //    160e: 3e 86         mv      a2, a5
            //    1610: c2 86         mv      a3, a6
            //    1612: 02 87         jr      a4
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction4", long.class, long.class, long.class, long.class, long.class),
                    new byte[]{
                            (byte) 0xba, (byte) 0x85,
                            (byte) 0x36, (byte) 0x85,
                            0x32, (byte) 0x87,
                            0x3e, (byte) 0x86,
                            (byte) 0xc2, (byte) 0x86,
                            0x02, (byte) 0x87
                    }
            );
            //0000000000001614 <NativeBridge_nativeGetJavaVM>:
            //    1614: 01 11         addi    sp, sp, -0x20
            //    1616: 06 ec         sd      ra, 0x18(sp)
            //    1618: 22 e8         sd      s0, 0x10(sp)
            //    161a: 00 10         addi    s0, sp, 0x20
            //    161c: 0c 61         ld      a1, 0x0(a0)
            //    161e: 03 b6 85 6d   ld      a2, 0x6d8(a1)
            //    1622: 23 34 04 fe   sd      zero, -0x18(s0)
            //    1626: 93 05 84 fe   addi    a1, s0, -0x18
            //    162a: 02 96         jalr    a2
            //    162c: 83 35 84 fe   ld      a1, -0x18(s0)
            //    1630: 33 35 a0 00   snez    a0, a0
            //    1634: 7d 15         addi    a0, a0, -0x1
            //    1636: 6d 8d         and     a0, a0, a1
            //    1638: 13 01 04 fe   addi    sp, s0, -0x20
            //    163c: e2 60         ld      ra, 0x18(sp)
            //    163e: 42 64         ld      s0, 0x10(sp)
            //    1640: 05 61         addi    sp, sp, 0x20
            //    1642: 82 80         ret
            result.put(
                    NativeBridge.class.getMethod("nativeGetJavaVM"),
                    new byte[]{
                            0x01, 0x11,
                            0x06, (byte) 0xec,
                            0x22, (byte) 0xe8,
                            0x00, 0x10,
                            0x0c, 0x61,
                            0x03, (byte) 0xb6, (byte) 0x85, 0x6d,
                            0x23, 0x34, 0x04, (byte) 0xfe,
                            (byte) 0x93, 0x05, (byte) 0x84, (byte) 0xfe,
                            0x02, (byte) 0x96,
                            (byte) 0x83, 0x35, (byte) 0x84, (byte) 0xfe,
                            0x33, 0x35, (byte) 0xa0, 0x00,
                            0x7d, 0x15,
                            0x6d, (byte) 0x8d,
                            0x13, 0x01, 0x04, (byte) 0xfe,
                            (byte) 0xe2, 0x60,
                            0x42, 0x64,
                            0x05, 0x61,
                            (byte) 0x82, (byte) 0x80
                    }
            );
            //0000000000001644 <NativeBridge_nativeSyscall>:
            //    1644: 82 62         ld      t0, 0x0(sp)
            //    1646: 46 83         mv      t1, a7
            //    1648: ba 85         mv      a1, a4
            //    164a: 36 85         mv      a0, a3
            //    164c: b2 88         mv      a7, a2
            //    164e: 3e 86         mv      a2, a5
            //    1650: c2 86         mv      a3, a6
            //    1652: 1a 87         mv      a4, t1
            //    1654: 96 87         mv      a5, t0
            //    1656: 73 00 00 00   ecall
            //    165a: 82 80         ret
            result.put(
                    NativeBridge.class.getMethod("nativeSyscall", int.class, long.class, long.class, long.class, long.class, long.class, long.class),
                    new byte[]{
                            (byte) 0x82, 0x62,
                            0x46, (byte) 0x83,
                            (byte) 0xba, (byte) 0x85,
                            (byte) 0x36, (byte) 0x85,
                            (byte) 0xb2, (byte) 0x88,
                            0x3e, (byte) 0x86,
                            (byte) 0xc2, (byte) 0x86,
                            0x1a, (byte) 0x87,
                            (byte) 0x96, (byte) 0x87,
                            0x73, 0x00, 0x00, 0x00,
                            (byte) 0x82, (byte) 0x80
                    }
            );
        } catch (NoSuchMethodException e) {
            ReflectHelper.unsafeThrow(e);
        }
        return Collections.unmodifiableMap(result);
    }

    @Override
    public int __NR_mprotect() {
        // mprotect riscv64 226
        return 226;
    }

    @Override
    public int __NR_memfd_create() {
        // memfd_create riscv64 279
        return 279;
    }
}
