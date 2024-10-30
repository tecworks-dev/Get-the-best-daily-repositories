package dev.tmpfs.libcoresyscall.core.impl.arch;

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import dev.tmpfs.libcoresyscall.core.impl.NativeBridge;
import dev.tmpfs.libcoresyscall.core.impl.ReflectHelper;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.BaseTrampolineCreator;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.ISyscallNumberTable;

public class TrampolineCreator_Mips64el extends BaseTrampolineCreator implements ISyscallNumberTable {

    private TrampolineCreator_Mips64el() {
    }

    public static final TrampolineCreator_Mips64el INSTANCE = new TrampolineCreator_Mips64el();

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
        //0000000000000748 <NativeBridge_breakpoint>:
        // 748:   0000000d        break
        return instructionsToBytes(0x0000000d);
    }

    @Override
    public Map<Method, byte[]> getNativeMethods() {
        HashMap<Method, byte[]> result = new HashMap<>();
        try {
            //0000000000000750 <NativeBridge_nativeClearCache>:
            // 750:   7c02083b        rdhwr   v0,hwr_synci_step
            // 754:   d840000e        beqzc   v0,790 <.LBB2_4>
            // 758:   00e6182d        daddu   v1,a3,a2
            // 75c:   00c3082b        sltu    at,a2,v1
            // 760:   d8200006        beqzc   at,77c <.LBB2_3>
            // 764:   00000000        nop
            //0000000000000768 <.LBB2_2>:
            // 768:   04df0000        synci   0(a2)
            // 76c:   00c2302d        daddu   a2,a2,v0
            // 770:   00c3082b        sltu    at,a2,v1
            // 774:   f83ffffc        bnezc   at,768 <.LBB2_2>
            // 778:   00000000        nop
            //000000000000077c <.LBB2_3>:
            // 77c:   0000000f        sync
            // 780:   ec200003        lapc    at,78c <.LBB2_3+0x10>
            // 784:   00200409        jr.hb   at
            // 788:   00000000        nop
            // 78c:   0000082d        move    at,zero
            //0000000000000790 <.LBB2_4>:
            // 790:   d81f0000        jrc     ra
            // 794:   00000000        nop
            result.put(
                    NativeBridge.class.getMethod("nativeClearCache", long.class, long.class),
                    instructionsToBytes(
                            0x7c02083b,
                            0xd840000e,
                            0x00e6182d,
                            0x00c3082b,
                            0xd8200006,
                            0x00000000,
                            0x04df0000,
                            0x00c2302d,
                            0x00c3082b,
                            0xf83ffffc,
                            0x00000000,
                            0x0000000f,
                            0xec200003,
                            0x00200409,
                            0x00000000,
                            0x0000082d,
                            0xd81f0000,
                            0x00000000
                    )
            );
            //0000000000000798 <NativeBridge_nativeCallPointerFunction0>:
            // 798:   67bdfff0        daddiu  sp,sp,-16
            // 79c:   ffbf0008        sd      ra,8(sp)
            // 7a0:   00c0c82d        move    t9,a2
            // 7a4:   f8190000        jalrc   t9
            // 7a8:   dfbf0008        ld      ra,8(sp)
            // 7ac:   03e00009        jr      ra
            // 7b0:   67bd0010        daddiu  sp,sp,16
            // 7b4:   00000000        nop
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction0", long.class),
                    instructionsToBytes(
                            0x67bdfff0,
                            0xffbf0008,
                            0x00c0c82d,
                            0xf8190000,
                            0xdfbf0008,
                            0x03e00009,
                            0x67bd0010,
                            0x00000000
                    )
            );
            //00000000000007b8 <NativeBridge_nativeCallPointerFunction1>:
            // 7b8:   67bdfff0        daddiu  sp,sp,-16
            // 7bc:   ffbf0008        sd      ra,8(sp)
            // 7c0:   00c0c82d        move    t9,a2
            // 7c4:   0320f809        jalr    t9
            // 7c8:   00e0202d        move    a0,a3
            // 7cc:   dfbf0008        ld      ra,8(sp)
            // 7d0:   03e00009        jr      ra
            // 7d4:   67bd0010        daddiu  sp,sp,16
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction1", long.class, long.class),
                    instructionsToBytes(
                            0x67bdfff0,
                            0xffbf0008,
                            0x00c0c82d,
                            0x0320f809,
                            0x00e0202d,
                            0xdfbf0008,
                            0x03e00009,
                            0x67bd0010
                    )
            );
            //00000000000007d8 <NativeBridge_nativeCallPointerFunction2>:
            // 7d8:   67bdfff0        daddiu  sp,sp,-16
            // 7dc:   ffbf0008        sd      ra,8(sp)
            // 7e0:   00e0202d        move    a0,a3
            // 7e4:   00c0c82d        move    t9,a2
            // 7e8:   0320f809        jalr    t9
            // 7ec:   0100282d        move    a1,a4
            // 7f0:   dfbf0008        ld      ra,8(sp)
            // 7f4:   03e00009        jr      ra
            // 7f8:   67bd0010        daddiu  sp,sp,16
            // 7fc:   00000000        nop
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction2", long.class, long.class, long.class),
                    instructionsToBytes(
                            0x67bdfff0,
                            0xffbf0008,
                            0x00e0202d,
                            0x00c0c82d,
                            0x0320f809,
                            0x0100282d,
                            0xdfbf0008,
                            0x03e00009,
                            0x67bd0010,
                            0x00000000
                    )
            );
            //0000000000000800 <NativeBridge_nativeCallPointerFunction3>:
            // 800:   67bdfff0        daddiu  sp,sp,-16
            // 804:   ffbf0008        sd      ra,8(sp)
            // 808:   00c0082d        move    at,a2
            // 80c:   00e0202d        move    a0,a3
            // 810:   0100282d        move    a1,a4
            // 814:   0020c82d        move    t9,at
            // 818:   0320f809        jalr    t9
            // 81c:   0120302d        move    a2,a5
            // 820:   dfbf0008        ld      ra,8(sp)
            // 824:   03e00009        jr      ra
            // 828:   67bd0010        daddiu  sp,sp,16
            // 82c:   00000000        nop
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction3", long.class, long.class, long.class, long.class),
                    instructionsToBytes(
                            0x67bdfff0,
                            0xffbf0008,
                            0x00c0082d,
                            0x00e0202d,
                            0x0100282d,
                            0x0020c82d,
                            0x0320f809,
                            0x0120302d,
                            0xdfbf0008,
                            0x03e00009,
                            0x67bd0010,
                            0x00000000
                    )
            );
            //0000000000000830 <NativeBridge_nativeCallPointerFunction4>:
            // 830:   67bdfff0        daddiu  sp,sp,-16
            // 834:   ffbf0008        sd      ra,8(sp)
            // 838:   00c0082d        move    at,a2
            // 83c:   00e0202d        move    a0,a3
            // 840:   0100282d        move    a1,a4
            // 844:   0120302d        move    a2,a5
            // 848:   0020c82d        move    t9,at
            // 84c:   0320f809        jalr    t9
            // 850:   0140382d        move    a3,a6
            // 854:   dfbf0008        ld      ra,8(sp)
            // 858:   03e00009        jr      ra
            // 85c:   67bd0010        daddiu  sp,sp,16
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction4", long.class, long.class, long.class, long.class, long.class),
                    instructionsToBytes(
                            0x67bdfff0,
                            0xffbf0008,
                            0x00c0082d,
                            0x00e0202d,
                            0x0100282d,
                            0x0120302d,
                            0x0020c82d,
                            0x0320f809,
                            0x0140382d,
                            0xdfbf0008,
                            0x03e00009,
                            0x67bd0010
                    )
            );
            //0000000000000860 <NativeBridge_nativeGetJavaVM>:
            // 860:   67bdfff0        daddiu  sp,sp,-16
            // 864:   ffbf0008        sd      ra,8(sp)
            // 868:   ffa00000        sd      zero,0(sp)
            // 86c:   dc810000        ld      at,0(a0)
            // 870:   dc3906d8        ld      t9,1752(at)
            // 874:   0320f809        jalr    t9
            // 878:   67a50000        daddiu  a1,sp,0
            // 87c:   00020800        sll     at,v0,0x0
            // 880:   dfa20000        ld      v0,0(sp)
            // 884:   dfbf0008        ld      ra,8(sp)
            // 888:   00411035        seleqz  v0,v0,at
            // 88c:   03e00009        jr      ra
            // 890:   67bd0010        daddiu  sp,sp,16
            result.put(
                    NativeBridge.class.getMethod("nativeGetJavaVM"),
                    instructionsToBytes(
                            0x67bdfff0,
                            0xffbf0008,
                            0xffa00000,
                            0xdc810000,
                            0xdc3906d8,
                            0x0320f809,
                            0x67a50000,
                            0x00020800,
                            0xdfa20000,
                            0xdfbf0008,
                            0x00411035,
                            0x03e00009,
                            0x67bd0010
                    )
            );
            //0000000000000710 <NativeBridge_nativeSyscall>:
            // 710:   dfa10000        ld      at,0(sp)
            // 714:   00c0102d        move    v0,a2
            // 718:   00e0202d        move    a0,a3
            // 71c:   0100282d        move    a1,a4
            // 720:   0120302d        move    a2,a5
            // 724:   0160402d        move    a4,a7
            // 728:   0140382d        move    a3,a6
            // 72c:   0020482d        move    a5,at
            // 730:   0000000c        syscall
            // 734:   0002082f        dnegu   at,v0
            // 738:   00471035        seleqz  v0,v0,a3
            // 73c:   00270837        selnez  at,at,a3
            // 740:   03e00009        jr      ra
            // 744:   00411025        or      v0,v0,at
            result.put(
                    NativeBridge.class.getMethod("nativeSyscall", int.class, long.class, long.class, long.class, long.class, long.class, long.class),
                    instructionsToBytes(
                            0xdfa10000,
                            0x00c0102d,
                            0x00e0202d,
                            0x0100282d,
                            0x0120302d,
                            0x0160402d,
                            0x0140382d,
                            0x0020482d,
                            0x0000000c,
                            0x0002082f,
                            0x00471035,
                            0x00270837,
                            0x03e00009,
                            0x00411025
                    )
            );
        } catch (NoSuchMethodException e) {
            ReflectHelper.unsafeThrow(e);
        }
        return Collections.unmodifiableMap(result);
    }

    @Override
    public int __NR_mprotect() {
        // mprotect mips64el n64 5000+10
        return 5010;
    }

    @Override
    public int __NR_memfd_create() {
        // memfd_create mips64el n64 5000+314
        return 5314;
    }
}
