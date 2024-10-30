package dev.tmpfs.libcoresyscall.core;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

import dev.tmpfs.libcoresyscall.core.impl.NativeBridge;
import libcore.io.Memory;

public class MemoryAccess {

    private MemoryAccess() {
        throw new AssertionError("no instances");
    }

    public static long getPageSize() {
        return NativeBridge.getPageSize();
    }

    // Since libcore.io.Memory is not part of public API, so we place a wrapper here.

    public static void peekByteArray(long address, byte[] dst, int offset, int byteCount) {
        Memory.peekByteArray(address, dst, offset, byteCount);
    }

    public static void pokeByteArray(long address, byte[] src, int offset, int byteCount) {
        Memory.pokeByteArray(address, src, offset, byteCount);
    }

    public static long peekLong(long address, boolean swap) {
        return Memory.peekLong(address, swap);
    }

    public static void pokeLong(long address, long value, boolean swap) {
        Memory.pokeLong(address, value, swap);
    }

    public static long peekLong(long address) {
        return Memory.peekLong(address, false);
    }

    public static void pokeLong(long address, long value) {
        Memory.pokeLong(address, value, false);
    }

    public static int peekInt(long address, boolean swap) {
        return Memory.peekInt(address, swap);
    }

    public static int peekInt(long address) {
        return Memory.peekInt(address, false);
    }

    public static void pokeInt(long address, int value, boolean swap) {
        Memory.pokeInt(address, value, swap);
    }

    public static void pokeInt(long address, int value) {
        Memory.pokeInt(address, value, false);
    }

    public static short peekByte(long address) {
        return Memory.peekByte(address);
    }

    /**
     * Peek a null-terminated string from the specified address.
     *
     * @param address the address of the string
     * @return the string
     */
    public static String peekCString(long address) {
        if (address == 0) {
            throw new NullPointerException("address is null");
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        long i = address;
        while (true) {
            byte b = Memory.peekByte(i++);
            if (b == 0) {
                break;
            }
            baos.write(b);
        }
        return new String(baos.toByteArray(), StandardCharsets.UTF_8);
    }

    public static void memset(long address, int c, long count) {
        if (address == 0) {
            throw new NullPointerException("address is null");
        }
        if (count < 0) {
            throw new IllegalArgumentException("count is negative");
        }
        if (count == 0) {
            return;
        }
        {
            // perform non-8-byte aligned head
            long align8Start = (address + 7) & ~7;
            if (address != align8Start) {
                // handle the first few bytes
                int n = (int) Math.min(align8Start - address, count);
                for (int i = 0; i < n; i++) {
                    Memory.pokeByte(address + i, (byte) c);
                }
                address = align8Start;
                count -= n;
            }
        }
        if (count >= 8) {
            // handle 8 bytes at a time
            long cccc = c & 0xff;
            cccc |= cccc << 8;
            cccc |= cccc << 16;
            cccc |= cccc << 32;
            long end = address + (count & ~7);
            for (; address < end; address += 8) {
                Memory.pokeLong(address, cccc, false);
            }
            count &= 7;
        }
        if (count > 0) {
            // handle the last few bytes
            for (int i = 0; i < count; i++) {
                Memory.pokeByte(address + i, (byte) c);
            }
        }
    }

}
