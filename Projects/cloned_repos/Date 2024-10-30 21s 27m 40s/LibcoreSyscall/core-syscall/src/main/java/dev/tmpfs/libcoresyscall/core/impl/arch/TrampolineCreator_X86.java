package dev.tmpfs.libcoresyscall.core.impl.arch;

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import dev.tmpfs.libcoresyscall.core.impl.NativeBridge;
import dev.tmpfs.libcoresyscall.core.impl.ReflectHelper;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.BaseTrampolineCreator;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.ISyscallNumberTable;

public class TrampolineCreator_X86 extends BaseTrampolineCreator implements ISyscallNumberTable {

    private TrampolineCreator_X86() {
    }

    public static final TrampolineCreator_X86 INSTANCE = new TrampolineCreator_X86();

    @Override
    public byte[] getPaddingInstruction() {
        // int3
        return new byte[]{(byte) 0xCC};
    }

    @Override
    public Map<Method, byte[]> getNativeMethods() {
        HashMap<Method, byte[]> result = new HashMap<>();
        try {
            //00001540 <NativeBridge_nativeClearCache>:
            //    1540: 55                            pushl   %ebp
            //    1541: 89 e5                         movl    %esp, %ebp
            //    1543: 83 e4 fc                      andl    $-0x4, %esp
            //    1546: 89 ec                         movl    %ebp, %esp
            //    1548: 5d                            popl    %ebp
            //    1549: c3                            retl
            //    154a: 90                            nop
            //    154b: 90                            nop
            result.put(
                    NativeBridge.class.getMethod("nativeClearCache", long.class, long.class),
                    new byte[]{
                            (byte) 0x55,
                            (byte) 0x89, (byte) 0xE5,
                            (byte) 0x83, (byte) 0xE4, (byte) 0xFC,
                            (byte) 0x89, (byte) 0xEC,
                            (byte) 0x5D,
                            (byte) 0xC3,
                            (byte) 0x90,
                            (byte) 0x90
                    }
            );
            //00001550 <NativeBridge_nativeCallPointerFunction0>:
            //    1550: 55                            pushl   %ebp
            //    1551: 89 e5                         movl    %esp, %ebp
            //    1553: 53                            pushl   %ebx
            //    1554: 83 e4 f0                      andl    $-0x10, %esp
            //    1557: 83 ec 10                      subl    $0x10, %esp
            //    155a: e8 00 00 00 00                calll   0x155f <NativeBridge_nativeCallPointerFunction0+0xf>
            //    155f: 5b                            popl    %ebx
            //    1560: 81 c3 d5 11 00 00             addl    $0x11d5, %ebx           # imm = 0x11D5
            //    1566: ff 55 10                      calll   *0x10(%ebp)
            //    1569: 31 d2                         xorl    %edx, %edx
            //    156b: 8d 65 fc                      leal    -0x4(%ebp), %esp
            //    156e: 5b                            popl    %ebx
            //    156f: 5d                            popl    %ebp
            //    1570: c3                            retl
            //    1571: 90                            nop
            //    1572: 90                            nop
            //    1573: 90                            nop
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction0", long.class),
                    new byte[]{
                            (byte) 0x55,
                            (byte) 0x89, (byte) 0xE5,
                            (byte) 0x53,
                            (byte) 0x83, (byte) 0xE4, (byte) 0xF0,
                            (byte) 0x83, (byte) 0xEC, (byte) 0x10,
                            (byte) 0xE8, 0x00, 0x00, 0x00, 0x00,
                            (byte) 0x5B,
                            (byte) 0x81, (byte) 0xC3, (byte) 0xD5, (byte) 0x11, 0x00, 0x00,
                            (byte) 0xFF, (byte) 0x55, 0x10,
                            (byte) 0x31, (byte) 0xD2,
                            (byte) 0x8D, 0x65, (byte) 0xFC,
                            (byte) 0x5B,
                            (byte) 0x5D,
                            (byte) 0xC3,
                            (byte) 0x90,
                            (byte) 0x90,
                            (byte) 0x90
                    }
            );
            //00001580 <NativeBridge_nativeCallPointerFunction1>:
            //    1580: 55                            pushl   %ebp
            //    1581: 89 e5                         movl    %esp, %ebp
            //    1583: 53                            pushl   %ebx
            //    1584: 83 e4 f0                      andl    $-0x10, %esp
            //    1587: 83 ec 10                      subl    $0x10, %esp
            //    158a: e8 00 00 00 00                calll   0x158f <NativeBridge_nativeCallPointerFunction1+0xf>
            //    158f: 5b                            popl    %ebx
            //    1590: 81 c3 a5 11 00 00             addl    $0x11a5, %ebx           # imm = 0x11A5
            //    1596: 8b 45 18                      movl    0x18(%ebp), %eax
            //    1599: 89 04 24                      movl    %eax, (%esp)
            //    159c: ff 55 10                      calll   *0x10(%ebp)
            //    159f: 31 d2                         xorl    %edx, %edx
            //    15a1: 8d 65 fc                      leal    -0x4(%ebp), %esp
            //    15a4: 5b                            popl    %ebx
            //    15a5: 5d                            popl    %ebp
            //    15a6: c3                            retl
            //    15a7: 90                            nop
            //    15a8: 90                            nop
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction1", long.class, long.class),
                    new byte[]{
                            (byte) 0x55,
                            (byte) 0x89, (byte) 0xE5,
                            (byte) 0x53,
                            (byte) 0x83, (byte) 0xE4, (byte) 0xF0,
                            (byte) 0x83, (byte) 0xEC, (byte) 0x10,
                            (byte) 0xE8, 0x00, 0x00, 0x00, 0x00,
                            (byte) 0x5B,
                            (byte) 0x81, (byte) 0xC3, (byte) 0xA5, (byte) 0x11, 0x00, 0x00,
                            (byte) 0x8B, 0x45, 0x18,
                            (byte) 0x89, 0x04, 0x24,
                            (byte) 0xFF, (byte) 0x55, 0x10,
                            (byte) 0x31, (byte) 0xD2,
                            (byte) 0x8D, 0x65, (byte) 0xFC,
                            (byte) 0x5B,
                            (byte) 0x5D,
                            (byte) 0xC3,
                            (byte) 0x90,
                            (byte) 0x90
                    }
            );
            //000015b0 <NativeBridge_nativeCallPointerFunction2>:
            //    15b0: 55                            pushl   %ebp
            //    15b1: 89 e5                         movl    %esp, %ebp
            //    15b3: 53                            pushl   %ebx
            //    15b4: 83 e4 f0                      andl    $-0x10, %esp
            //    15b7: 83 ec 10                      subl    $0x10, %esp
            //    15ba: e8 00 00 00 00                calll   0x15bf <NativeBridge_nativeCallPointerFunction2+0xf>
            //    15bf: 5b                            popl    %ebx
            //    15c0: 81 c3 75 11 00 00             addl    $0x1175, %ebx           # imm = 0x1175
            //    15c6: 83 ec 08                      subl    $0x8, %esp
            //    15c9: ff 75 20                      pushl   0x20(%ebp)
            //    15cc: ff 75 18                      pushl   0x18(%ebp)
            //    15cf: ff 55 10                      calll   *0x10(%ebp)
            //    15d2: 83 c4 10                      addl    $0x10, %esp
            //    15d5: 31 d2                         xorl    %edx, %edx
            //    15d7: 8d 65 fc                      leal    -0x4(%ebp), %esp
            //    15da: 5b                            popl    %ebx
            //    15db: 5d                            popl    %ebp
            //    15dc: c3                            retl
            //    15dd: 90                            nop
            //    15de: 90                            nop
            //    15df: 90                            nop
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction2", long.class, long.class, long.class),
                    new byte[]{
                            (byte) 0x55,
                            (byte) 0x89, (byte) 0xE5,
                            (byte) 0x53,
                            (byte) 0x83, (byte) 0xE4, (byte) 0xF0,
                            (byte) 0x83, (byte) 0xEC, (byte) 0x10,
                            (byte) 0xE8, 0x00, 0x00, 0x00, 0x00,
                            (byte) 0x5B,
                            (byte) 0x81, (byte) 0xC3, (byte) 0x75, (byte) 0x11, 0x00, 0x00,
                            (byte) 0x83, (byte) 0xEC, 0x08,
                            (byte) 0xFF, 0x75, 0x20,
                            (byte) 0xFF, 0x75, 0x18,
                            (byte) 0xFF, (byte) 0x55, 0x10,
                            (byte) 0x83, (byte) 0xC4, 0x10,
                            (byte) 0x31, (byte) 0xD2,
                            (byte) 0x8D, 0x65, (byte) 0xFC,
                            (byte) 0x5B,
                            (byte) 0x5D,
                            (byte) 0xC3,
                            (byte) 0x90,
                            (byte) 0x90,
                            (byte) 0x90
                    }
            );
            //000015e0 <NativeBridge_nativeCallPointerFunction3>:
            //    15e0: 55                            pushl   %ebp
            //    15e1: 89 e5                         movl    %esp, %ebp
            //    15e3: 53                            pushl   %ebx
            //    15e4: 83 e4 f0                      andl    $-0x10, %esp
            //    15e7: 83 ec 10                      subl    $0x10, %esp
            //    15ea: e8 00 00 00 00                calll   0x15ef <NativeBridge_nativeCallPointerFunction3+0xf>
            //    15ef: 5b                            popl    %ebx
            //    15f0: 81 c3 45 11 00 00             addl    $0x1145, %ebx           # imm = 0x1145
            //    15f6: 83 ec 04                      subl    $0x4, %esp
            //    15f9: ff 75 28                      pushl   0x28(%ebp)
            //    15fc: ff 75 20                      pushl   0x20(%ebp)
            //    15ff: ff 75 18                      pushl   0x18(%ebp)
            //    1602: ff 55 10                      calll   *0x10(%ebp)
            //    1605: 83 c4 10                      addl    $0x10, %esp
            //    1608: 31 d2                         xorl    %edx, %edx
            //    160a: 8d 65 fc                      leal    -0x4(%ebp), %esp
            //    160d: 5b                            popl    %ebx
            //    160e: 5d                            popl    %ebp
            //    160f: c3                            retl
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction3", long.class, long.class, long.class, long.class),
                    new byte[]{
                            (byte) 0x55,
                            (byte) 0x89, (byte) 0xE5,
                            (byte) 0x53,
                            (byte) 0x83, (byte) 0xE4, (byte) 0xF0,
                            (byte) 0x83, (byte) 0xEC, (byte) 0x10,
                            (byte) 0xE8, 0x00, 0x00, 0x00, 0x00,
                            (byte) 0x5B,
                            (byte) 0x81, (byte) 0xC3, (byte) 0x45, (byte) 0x11, 0x00, 0x00,
                            (byte) 0x83, (byte) 0xEC, 0x04,
                            (byte) 0xFF, 0x75, 0x28,
                            (byte) 0xFF, 0x75, 0x20,
                            (byte) 0xFF, 0x75, 0x18,
                            (byte) 0xFF, (byte) 0x55, 0x10,
                            (byte) 0x83, (byte) 0xC4, 0x10,
                            (byte) 0x31, (byte) 0xD2,
                            (byte) 0x8D, 0x65, (byte) 0xFC,
                            (byte) 0x5B,
                            (byte) 0x5D,
                            (byte) 0xC3
                    }
            );
            //00001610 <NativeBridge_nativeCallPointerFunction4>:
            //    1610: 55                            pushl   %ebp
            //    1611: 89 e5                         movl    %esp, %ebp
            //    1613: 53                            pushl   %ebx
            //    1614: 83 e4 f0                      andl    $-0x10, %esp
            //    1617: 83 ec 10                      subl    $0x10, %esp
            //    161a: e8 00 00 00 00                calll   0x161f <NativeBridge_nativeCallPointerFunction4+0xf>
            //    161f: 5b                            popl    %ebx
            //    1620: 81 c3 15 11 00 00             addl    $0x1115, %ebx           # imm = 0x1115
            //    1626: ff 75 30                      pushl   0x30(%ebp)
            //    1629: ff 75 28                      pushl   0x28(%ebp)
            //    162c: ff 75 20                      pushl   0x20(%ebp)
            //    162f: ff 75 18                      pushl   0x18(%ebp)
            //    1632: ff 55 10                      calll   *0x10(%ebp)
            //    1635: 83 c4 10                      addl    $0x10, %esp
            //    1638: 31 d2                         xorl    %edx, %edx
            //    163a: 8d 65 fc                      leal    -0x4(%ebp), %esp
            //    163d: 5b                            popl    %ebx
            //    163e: 5d                            popl    %ebp
            //    163f: c3                            retl
            result.put(
                    NativeBridge.class.getMethod("nativeCallPointerFunction4", long.class, long.class, long.class, long.class, long.class),
                    new byte[]{
                            (byte) 0x55,
                            (byte) 0x89, (byte) 0xE5,
                            (byte) 0x53,
                            (byte) 0x83, (byte) 0xE4, (byte) 0xF0,
                            (byte) 0x83, (byte) 0xEC, (byte) 0x10,
                            (byte) 0xE8, 0x00, 0x00, 0x00, 0x00,
                            (byte) 0x5B,
                            (byte) 0x81, (byte) 0xC3, (byte) 0x15, (byte) 0x11, 0x00, 0x00,
                            (byte) 0xFF, 0x75, 0x30,
                            (byte) 0xFF, 0x75, 0x28,
                            (byte) 0xFF, 0x75, 0x20,
                            (byte) 0xFF, 0x75, 0x18,
                            (byte) 0xFF, (byte) 0x55, 0x10,
                            (byte) 0x83, (byte) 0xC4, 0x10,
                            (byte) 0x31, (byte) 0xD2,
                            (byte) 0x8D, 0x65, (byte) 0xFC,
                            (byte) 0x5B,
                            (byte) 0x5D,
                            (byte) 0xC3
                    }
            );
            //00001640 <NativeBridge_nativeGetJavaVM>:
            //    1640: 55                            pushl   %ebp
            //    1641: 89 e5                         movl    %esp, %ebp
            //    1643: 53                            pushl   %ebx
            //    1644: 83 e4 f0                      andl    $-0x10, %esp
            //    1647: 83 ec 10                      subl    $0x10, %esp
            //    164a: e8 00 00 00 00                calll   0x164f <NativeBridge_nativeGetJavaVM+0xf>
            //    164f: 5b                            popl    %ebx
            //    1650: 81 c3 e5 10 00 00             addl    $0x10e5, %ebx           # imm = 0x10E5
            //    1656: 8b 45 08                      movl    0x8(%ebp), %eax
            //    1659: c7 04 24 00 00 00 00          movl    $0x0, (%esp)
            //    1660: 8b 08                         movl    (%eax), %ecx
            //    1662: 83 ec 08                      subl    $0x8, %esp
            //    1665: 8d 54 24 08                   leal    0x8(%esp), %edx
            //    1669: 52                            pushl   %edx
            //    166a: 50                            pushl   %eax
            //    166b: ff 91 6c 03 00 00             calll   *0x36c(%ecx)
            //    1671: 83 c4 10                      addl    $0x10, %esp
            //    1674: 89 c1                         movl    %eax, %ecx
            //    1676: b8 00 00 00 00                movl    $0x0, %eax
            //    167b: 85 c9                         testl   %ecx, %ecx
            //    167d: 75 03                         jne     0x1682 <NativeBridge_nativeGetJavaVM+0x42>
            //    167f: 8b 04 24                      movl    (%esp), %eax
            //    1682: 31 d2                         xorl    %edx, %edx
            //    1684: 8d 65 fc                      leal    -0x4(%ebp), %esp
            //    1687: 5b                            popl    %ebx
            //    1688: 5d                            popl    %ebp
            //    1689: c3                            retl
            //    168a: 90                            nop
            //    168b: 90                            nop
            result.put(
                    NativeBridge.class.getMethod("nativeGetJavaVM"),
                    new byte[]{
                            (byte) 0x55,
                            (byte) 0x89, (byte) 0xE5,
                            (byte) 0x53,
                            (byte) 0x83, (byte) 0xE4, (byte) 0xF0,
                            (byte) 0x83, (byte) 0xEC, (byte) 0x10,
                            (byte) 0xE8, 0x00, 0x00, 0x00, 0x00,
                            (byte) 0x5B,
                            (byte) 0x81, (byte) 0xC3, (byte) 0xE5, (byte) 0x10, 0x00, 0x00,
                            (byte) 0x8B, 0x45, 0x08,
                            (byte) 0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00,
                            (byte) 0x8B, 0x08,
                            (byte) 0x83, (byte) 0xEC, 0x08,
                            (byte) 0x8D, 0x54, 0x24, 0x08,
                            (byte) 0x52,
                            (byte) 0x50,
                            (byte) 0xFF, (byte) 0x91, 0x6C, 0x03, 0x00, 0x00,
                            (byte) 0x83, (byte) 0xC4, 0x10,
                            (byte) 0x89, (byte) 0xC1,
                            (byte) 0xB8, 0x00, 0x00, 0x00, 0x00,
                            (byte) 0x85, (byte) 0xC9,
                            (byte) 0x75, 0x03,
                            (byte) 0x8B, 0x04, 0x24,
                            (byte) 0x31, (byte) 0xD2,
                            (byte) 0x8D, 0x65, (byte) 0xFC,
                            (byte) 0x5B,
                            (byte) 0x5D,
                            (byte) 0xC3,
                            (byte) 0x90,
                            (byte) 0x90
                    }
            );
            //00001200 <NativeBridge_nativeSyscall>:
            //    1200:       55                      push   ebp
            //    1201:       89 e5                   mov    ebp,esp
            //    1203:       57                      push   edi
            //    1204:       56                      push   esi
            //    1205:       83 e4 fc                and    esp,0xfffffffc
            //    1208:       83 ec 10                sub    esp,0x10
            //    120b:       8b 4d 1c                mov    ecx,DWORD PTR [ebp+0x1c]
            //    120e:       8b 55 24                mov    edx,DWORD PTR [ebp+0x24]
            //    1211:       8b 75 2c                mov    esi,DWORD PTR [ebp+0x2c]
            //    1214:       8b 7d 34                mov    edi,DWORD PTR [ebp+0x34]
            //    1217:       8b 45 14                mov    eax,DWORD PTR [ebp+0x14]
            //    121a:       89 44 24 04             mov    DWORD PTR [esp+0x4],eax
            //    121e:       8b 45 3c                mov    eax,DWORD PTR [ebp+0x3c]
            //    1221:       89 44 24 08             mov    DWORD PTR [esp+0x8],eax
            //    1225:       8b 45 10                mov    eax,DWORD PTR [ebp+0x10]
            //    1228:       89 44 24 0c             mov    DWORD PTR [esp+0xc],eax
            //    122c:       8d 44 24 04             lea    eax,[esp+0x4]
            //    1230:       55                      push   ebp
            //    1231:       53                      push   ebx
            //    1232:       8b 68 04                mov    ebp,DWORD PTR [eax+0x4]
            //    1235:       8b 18                   mov    ebx,DWORD PTR [eax]
            //    1237:       8b 40 08                mov    eax,DWORD PTR [eax+0x8]
            //    123a:       cd 80                   int    0x80
            //    123c:       5b                      pop    ebx
            //    123d:       5d                      pop    ebp
            //    123e:       89 c2                   mov    edx,eax
            //    1240:       c1 fa 1f                sar    edx,0x1f
            //    1243:       8d 65 f8                lea    esp,[ebp-0x8]
            //    1246:       5e                      pop    esi
            //    1247:       5f                      pop    edi
            //    1248:       5d                      pop    ebp
            //    1249:       c3                      ret
            result.put(
                    NativeBridge.class.getMethod("nativeSyscall", int.class, long.class, long.class, long.class, long.class, long.class, long.class),
                    new byte[]{
                            (byte) 0x55,
                            (byte) 0x89, (byte) 0xE5,
                            (byte) 0x57,
                            (byte) 0x56,
                            (byte) 0x83, (byte) 0xE4, (byte) 0xFC,
                            (byte) 0x83, (byte) 0xEC, 0x10,
                            (byte) 0x8B, 0x4D, 0x1C,
                            (byte) 0x8B, 0x55, 0x24,
                            (byte) 0x8B, 0x75, 0x2C,
                            (byte) 0x8B, 0x7D, 0x34,
                            (byte) 0x8B, 0x45, 0x14,
                            (byte) 0x89, 0x44, 0x24, 0x04,
                            (byte) 0x8B, 0x45, 0x3C,
                            (byte) 0x89, 0x44, 0x24, 0x08,
                            (byte) 0x8B, 0x45, 0x10,
                            (byte) 0x89, 0x44, 0x24, 0x0C,
                            (byte) 0x8D, 0x44, 0x24, 0x04,
                            (byte) 0x55,
                            (byte) 0x53,
                            (byte) 0x8B, 0x68, 0x04,
                            (byte) 0x8B, 0x18,
                            (byte) 0x8B, 0x40, 0x08,
                            (byte) 0xCD, (byte) 0x80,
                            (byte) 0x5B,
                            (byte) 0x5D,
                            (byte) 0x89, (byte) 0xC2,
                            (byte) 0xC1, (byte) 0xFA, 0x1F,
                            (byte) 0x8D, 0x65, (byte) 0xF8,
                            (byte) 0x5E,
                            (byte) 0x5F,
                            (byte) 0x5D,
                            (byte) 0xC3
                    }
            );
        } catch (NoSuchMethodException e) {
            ReflectHelper.unsafeThrow(e);
        }
        return Collections.unmodifiableMap(result);
    }

    @Override
    public int __NR_mprotect() {
        // __NR_mprotect x86 125
        return 125;
    }

    @Override
    public int __NR_memfd_create() {
        // __NR_memfd_create x86 356
        return 356;
    }
}
