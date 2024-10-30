package dev.tmpfs.libcoresyscall.core.impl.arch;

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import dev.tmpfs.libcoresyscall.core.impl.NativeBridge;
import dev.tmpfs.libcoresyscall.core.impl.ReflectHelper;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.BaseTrampolineCreator;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.ISyscallNumberTable;

public class TrampolineCreator_Arm extends BaseTrampolineCreator implements ISyscallNumberTable {

    private TrampolineCreator_Arm() {
    }

    public static final TrampolineCreator_Arm INSTANCE = new TrampolineCreator_Arm();

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
        //0000153c <NativeBridge_breakpoint>:
        //    153c: e1200070      bkpt    #0x0
        return instructionsToBytes(0xe1200070);
    }

    @Override
    public Map<Method, byte[]> getNativeMethods() {
        HashMap<Method, byte[]> result = new HashMap<>();
        try {
            //000011b8 <NativeBridge_nativeClearCache>:
            //    11b8: e92d4c80      push    {r7, r10, r11, lr}
            //    11bc: e28db008      add     r11, sp, #8
            //    11c0: e59b1008      ldr     r1, [r11, #0x8]
            //    11c4: e3007002      movw    r7, #0x2
            //    11c8: e1a00002      mov     r0, r2
            //    11cc: e340700f      movt    r7, #0xf
            //    11d0: e0811002      add     r1, r1, r2
            //    11d4: e3a02000      mov     r2, #0
            //    11d8: ef000000      svc     #0x0
            //    11dc: e3500000      cmp     r0, #0
            //    11e0: 08bd8c80      popeq   {r7, r10, r11, pc}
            //    11e4: e7ffdefe      trap
            result.put(
                    NativeBridge.class.getMethod("nativeClearCache", long.class, long.class),
                    instructionsToBytes(
                            0xe92d4c80,
                            0xe28db008,
                            0xe59b1008,
                            0xe3007002,
                            0xe1a00002,
                            0xe340700f,
                            0xe0811002,
                            0xe3a02000,
                            0xef000000,
                            0xe3500000,
                            0x08bd8c80,
                            0xe7ffdefe
                    )
            );
            //0000142c <NativeBridge_nativeCallPointerFunction0>:
            //    142c: e92d4800      push    {r11, lr}
            //    1430: e1a0b00d      mov     r11, sp
            //    1434: e12fff32      blx     r2
            //    1438: e3a01000      mov     r1, #0
            //    143c: e8bd8800      pop     {r11, pc}
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction0", long.class),
                    instructionsToBytes(
                            0xe92d4800,
                            0xe1a0b00d,
                            0xe12fff32,
                            0xe3a01000,
                            0xe8bd8800
                    )
            );
            //00001440 <NativeBridge_nativeCallPointerFunction1>:
            //    1440: e92d4800      push    {r11, lr}
            //    1444: e1a0b00d      mov     r11, sp
            //    1448: e59b0008      ldr     r0, [r11, #0x8]
            //    144c: e12fff32      blx     r2
            //    1450: e3a01000      mov     r1, #0
            //    1454: e8bd8800      pop     {r11, pc}
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction1", long.class, long.class),
                    instructionsToBytes(
                            0xe92d4800,
                            0xe1a0b00d,
                            0xe59b0008,
                            0xe12fff32,
                            0xe3a01000,
                            0xe8bd8800
                    )
            );
            //00001458 <NativeBridge_nativeCallPointerFunction2>:
            //    1458: e92d4800      push    {r11, lr}
            //    145c: e1a0b00d      mov     r11, sp
            //    1460: e59b0008      ldr     r0, [r11, #0x8]
            //    1464: e59b1010      ldr     r1, [r11, #0x10]
            //    1468: e12fff32      blx     r2
            //    146c: e3a01000      mov     r1, #0
            //    1470: e8bd8800      pop     {r11, pc}
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction2", long.class, long.class, long.class),
                    instructionsToBytes(
                            0xe92d4800,
                            0xe1a0b00d,
                            0xe59b0008,
                            0xe59b1010,
                            0xe12fff32,
                            0xe3a01000,
                            0xe8bd8800
                    )
            );
            //00001474 <NativeBridge_nativeCallPointerFunction3>:
            //    1474: e92d4800      push    {r11, lr}
            //    1478: e1a0b00d      mov     r11, sp
            //    147c: e1a03002      mov     r3, r2
            //    1480: e59b0008      ldr     r0, [r11, #0x8]
            //    1484: e59b1010      ldr     r1, [r11, #0x10]
            //    1488: e59b2018      ldr     r2, [r11, #0x18]
            //    148c: e12fff33      blx     r3
            //    1490: e3a01000      mov     r1, #0
            //    1494: e8bd8800      pop     {r11, pc}
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction3", long.class, long.class, long.class, long.class),
                    instructionsToBytes(
                            0xe92d4800,
                            0xe1a0b00d,
                            0xe1a03002,
                            0xe59b0008,
                            0xe59b1010,
                            0xe59b2018,
                            0xe12fff33,
                            0xe3a01000,
                            0xe8bd8800
                    )
            );
            //00001498 <NativeBridge_nativeCallPointerFunction4>:
            //    1498: e92d4800      push    {r11, lr}
            //    149c: e1a0b00d      mov     r11, sp
            //    14a0: e1a0c002      mov     r12, r2
            //    14a4: e59b0008      ldr     r0, [r11, #0x8]
            //    14a8: e59b1010      ldr     r1, [r11, #0x10]
            //    14ac: e59b2018      ldr     r2, [r11, #0x18]
            //    14b0: e59b3020      ldr     r3, [r11, #0x20]
            //    14b4: e12fff3c      blx     r12
            //    14b8: e3a01000      mov     r1, #0
            //    14bc: e8bd8800      pop     {r11, pc}
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction4", long.class, long.class, long.class, long.class, long.class),
                    instructionsToBytes(
                            0xe92d4800,
                            0xe1a0b00d,
                            0xe1a0c002,
                            0xe59b0008,
                            0xe59b1010,
                            0xe59b2018,
                            0xe59b3020,
                            0xe12fff3c,
                            0xe3a01000,
                            0xe8bd8800
                    )
            );
            //000014c0 <NativeBridge_nativeGetJavaVM>:
            //    14c0: e92d4c10      push    {r4, r10, r11, lr}
            //    14c4: e28db008      add     r11, sp, #8
            //    14c8: e24dd008      sub     sp, sp, #8
            //    14cc: e5901000      ldr     r1, [r0]
            //    14d0: e3a04000      mov     r4, #0
            //    14d4: e58d4004      str     r4, [sp, #0x4]
            //    14d8: e591236c      ldr     r2, [r1, #0x36c]
            //    14dc: e28d1004      add     r1, sp, #4
            //    14e0: e12fff32      blx     r2
            //    14e4: e59d1004      ldr     r1, [sp, #0x4]
            //    14e8: e3500000      cmp     r0, #0
            //    14ec: 11a01004      movne   r1, r4
            //    14f0: e1a00001      mov     r0, r1
            //    14f4: e3a01000      mov     r1, #0
            //    14f8: e24bd008      sub     sp, r11, #8
            //    14fc: e8bd8c10      pop     {r4, r10, r11, pc}
            result.put(
                    NativeBridge.class.getMethod("nativeGetJavaVM"),
                    instructionsToBytes(

                    )
            );
            //00001500 <NativeBridge_nativeSyscall>:
            //    1500: e92d4830      push    {r4, r5, r11, lr}
            //    1504: e28db008      add     r11, sp, #8
            //    1508: e59b0008      ldr     r0, [r11, #0x8]
            //    150c: e1a0c002      mov     r12, r2
            //    1510: e59b1010      ldr     r1, [r11, #0x10]
            //    1514: e59b2018      ldr     r2, [r11, #0x18]
            //    1518: e59b3020      ldr     r3, [r11, #0x20]
            //    151c: e59b4028      ldr     r4, [r11, #0x28]
            //    1520: e59b5030      ldr     r5, [r11, #0x30]
            //    1524: e52d7004      str     r7, [sp, #-0x4]!
            //    1528: e1a0700c      mov     r7, r12
            //    152c: ef000000      svc     #0x0
            //    1530: e49d7004      ldr     r7, [sp], #4
            //    1534: e1a01fc0      asr     r1, r0, #31
            //    1538: e8bd8830      pop     {r4, r5, r11, pc}
            result.put(
                    NativeBridge.class.getMethod("nativeSyscall", int.class, long.class, long.class, long.class, long.class, long.class, long.class),
                    instructionsToBytes(
                            0xe92d4830,
                            0xe28db008,
                            0xe59b0008,
                            0xe1a0c002,
                            0xe59b1010,
                            0xe59b2018,
                            0xe59b3020,
                            0xe59b4028,
                            0xe59b5030,
                            0xe52d7004,
                            0xe1a0700c,
                            0xef000000,
                            0xe49d7004,
                            0xe1a01fc0,
                            0xe8bd8830
                    )
            );
        } catch (NoSuchMethodException e) {
            ReflectHelper.unsafeThrow(e);
        }
        return Collections.unmodifiableMap(result);
    }

    @Override
    public int __NR_mprotect() {
        // mprotect arm: 125
        return 125;
    }

    @Override
    public int __NR_memfd_create() {
        // memfd_create arm: 385
        return 385;
    }
}
