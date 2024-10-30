package dev.tmpfs.libcoresyscall.core;

import android.system.ErrnoException;
import android.system.Os;
import android.system.OsConstants;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

import dev.tmpfs.libcoresyscall.core.impl.ReflectHelper;

public class MemoryAllocator {

    private MemoryAllocator() {
        throw new AssertionError("no instances");
    }

    private static final long PAGE_SIZE = (int) MemoryAccess.getPageSize();

    // We use 16 bytes alignment for the allocated memory.
    // Use call 16 bytes as a unit.
    private static final int UNIT_SIZE = 16;
    private static final int UNIT_PER_PAGE = (int) (PAGE_SIZE / UNIT_SIZE);

    // The memory block is big, and we can't allocate it in one page.
    private static class DirectPageMemory implements IAllocatedMemory {

        private volatile long mAddress;
        private volatile long mSize;
        private volatile long mAllocatedSize;

        private DirectPageMemory(long address, long size, long allocatedSize) {
            mAddress = address;
            mSize = size;
            mAllocatedSize = allocatedSize;
        }

        @Override
        public long getAddress() {
            return mAddress;
        }

        @Override
        public long getSize() {
            return mSize;
        }

        @Override
        public synchronized void free() {
            if (mAddress != 0) {
                freeDirectPageMemory(mAddress, mAllocatedSize);
                mAddress = 0;
                mSize = 0;
                mAllocatedSize = 0;
            }
        }

        @Override
        public void close() {
            free();
        }

        @Override
        public boolean isFreed() {
            return mAddress == 0;
        }
    }

    // The memory block is not big, and we can split it into units.
    private static class UnitMemory implements IAllocatedMemory {

        private volatile long mAddress;
        private volatile long mSize;
        private volatile long mAllocatedSize;

        private UnitMemory(long address, long size, long allocatedSize) {
            mAddress = address;
            mSize = size;
            mAllocatedSize = allocatedSize;
        }

        @Override
        public long getAddress() {
            return mAddress;
        }

        @Override
        public long getSize() {
            return mSize;
        }

        @Override
        public synchronized void free() {
            if (mAddress != 0) {
                freeUnitMemory(mAddress, mAllocatedSize);
                mAddress = 0;
                mSize = 0;
                mAllocatedSize = 0;
            }
        }

        @Override
        public void close() {
            free();
        }

        @Override
        public boolean isFreed() {
            return mAddress == 0;
        }
    }

    private static DirectPageMemory allocateDirectPageMemory(long requestedSize) {
        // align the size to page size.
        long alignedSize = (requestedSize + PAGE_SIZE - 1) & -PAGE_SIZE;
        final int MAP_ANONYMOUS = 0x20;
        try {
            long address = Os.mmap(0, alignedSize, OsConstants.PROT_READ | OsConstants.PROT_WRITE,
                    OsConstants.MAP_PRIVATE | MAP_ANONYMOUS, null, 0);
            if (address == 0) {
                throw new AssertionError("mmap failed with size " + alignedSize + ", but no errno");
            }
            return new DirectPageMemory(address, requestedSize, alignedSize);
        } catch (ErrnoException e) {
            throw ReflectHelper.unsafeThrow(e);
        }
    }

    private static void freeDirectPageMemory(long address, long allocatedSize) {
        try {
            Os.munmap(address, allocatedSize);
        } catch (ErrnoException e) {
            throw ReflectHelper.unsafeThrow(e);
        }
    }

    private static class PageUnitInfo {
        // The address of the page.
        public long address;
        // Allocated units, sorted, [unit index, unit count].
        // The unit index is the index of first unit, that is (address - page address) / UNIT_SIZE.
        public ArrayList<int[]> allocatedUnits = new ArrayList<>();

        public PageUnitInfo(long address) {
            this.address = address;
        }

    }

    private static final Object sUnitAllocLock = new Object();
    // A list of memory page, sorted by address.
    private static final ArrayList<PageUnitInfo> sUnitMemoryPageList = new ArrayList<>();

    /**
     * Find a memory page that has enough free units. If not found, return null.
     */
    private static UnitMemory tryAllocateMemoryUnitLocked(int unitCount, int requestedSize) {
        for (PageUnitInfo page : sUnitMemoryPageList) {
            if (page.allocatedUnits.isEmpty()) {
                // the page is empty, allocate the first unit.
                if (unitCount <= UNIT_PER_PAGE) {
                    page.allocatedUnits.add(new int[]{0, unitCount});
                    long address = page.address;
                    return new UnitMemory(address, requestedSize, (long) unitCount * UNIT_SIZE);
                }
            } else {
                // find a page that has enough free units.
                // find first in gap, and then tail.
                for (int gapIndex = 0; gapIndex < page.allocatedUnits.size(); gapIndex++) {
                    int[] thisUnit = page.allocatedUnits.get(gapIndex);
                    int[] nextUnit = gapIndex + 1 < page.allocatedUnits.size() ?
                            page.allocatedUnits.get(gapIndex + 1) : null;
                    int gapStart = thisUnit[0] + thisUnit[1];
                    int gapEnd = (nextUnit != null) ? nextUnit[0] : UNIT_PER_PAGE;
                    // check if the gap is enough.
                    if (gapEnd - gapStart >= unitCount) {
                        page.allocatedUnits.add(gapIndex, new int[]{gapStart, unitCount});
                        long address = page.address + (long) gapStart * UNIT_SIZE;
                        return new UnitMemory(address, requestedSize, (long) unitCount * UNIT_SIZE);
                    }
                }
            }
        }
        return null;
    }

    private static void brkNewPageForUnitAllocationLocked() {
        // We need to allocate a new page for unit allocation.
        final int MAP_ANONYMOUS = 0x20;
        try {
            long address = Os.mmap(0, PAGE_SIZE, OsConstants.PROT_READ | OsConstants.PROT_WRITE,
                    OsConstants.MAP_PRIVATE | MAP_ANONYMOUS, null, 0);
            if (address == 0) {
                throw new AssertionError("mmap failed with size " + PAGE_SIZE + ", but no errno");
            }
            synchronized (sUnitAllocLock) {
                sUnitMemoryPageList.add(new PageUnitInfo(address));
            }
        } catch (ErrnoException e) {
            throw ReflectHelper.unsafeThrow(e);
        }
    }

    private static UnitMemory allocateUnitMemory(int requestedSize) {
        if (requestedSize + UNIT_SIZE >= PAGE_SIZE) {
            // should not happen.
            throw new AssertionError("requested size is too big");
        }
        // align the size to unit size.
        final int allocateSizeBytes = (requestedSize + UNIT_SIZE - 1) & -UNIT_SIZE;
        final int allocateUnitCount = Math.max(allocateSizeBytes / UNIT_SIZE, 1);
        synchronized (sUnitAllocLock) {
            UnitMemory memory = tryAllocateMemoryUnitLocked(allocateUnitCount, requestedSize);
            if (memory != null) {
                return memory;
            }
            brkNewPageForUnitAllocationLocked();
            memory = tryAllocateMemoryUnitLocked(allocateUnitCount, requestedSize);
            if (memory != null) {
                return memory;
            }
            throw new AssertionError("failed to allocate unit memory, this should not happen");
        }
    }

    private static void freeUnitMemory(long address, long allocatedSize) {
        synchronized (sUnitAllocLock) {
            // find the page that contains the address.
            PageUnitInfo page = null;
            for (PageUnitInfo info : sUnitMemoryPageList) {
                if (address >= info.address && address < info.address + PAGE_SIZE) {
                    page = info;
                    break;
                }
            }
            if (page == null) {
                throw new AssertionError("invalid address to free");
            }
            // find the unit index.
            final int unitIndex = (int) ((address - page.address) / UNIT_SIZE);
            // remove the unit from the allocated list.
            for (int i = 0; i < page.allocatedUnits.size(); i++) {
                int[] unit = page.allocatedUnits.get(i);
                if (unit[0] == unitIndex) {
                    page.allocatedUnits.remove(i);
                    break;
                }
            }
            // do not free the page, keep the logic simple.
        }
    }

    /**
     * Allocate a memory block with the specified size. It's caller's responsibility to free the memory block.
     *
     * @param size   the size of the memory block
     * @param zeroed whether to zero the memory block
     * @return the allocated memory block
     */
    public static IAllocatedMemory allocate(long size, boolean zeroed) {
        // if requested size is 75% of page size, we allocate directly.
        if (size >= PAGE_SIZE * 3 / 4) {
            // anonymous memory maps are zeroed by default.
            return allocateDirectPageMemory(size);
        } else {
            IAllocatedMemory mem = allocateUnitMemory((int) size);
            // it may be used multiple times, so zero it here.
            if (zeroed) {
                MemoryAccess.memset(mem.getAddress(), 0, size);
            }
            return mem;
        }
    }

    /**
     * Allocate a memory block with the specified size. It's caller's responsibility to free the memory block.
     *
     * @param size the size of the memory block
     * @return the allocated memory block
     */
    public static IAllocatedMemory allocate(long size) {
        return allocate(size, false);
    }

    /**
     * Allocate a memory block and copy the specified bytes to it.
     *
     * @param bytes  the bytes to copy
     * @param offset the offset in the bytes
     * @param length the length of the bytes
     * @return the allocated memory block
     */
    public static IAllocatedMemory copyBytes(byte[] bytes, int offset, int length) {
        IAllocatedMemory memory = allocate(length);
        MemoryAccess.pokeByteArray(memory.getAddress(), bytes, offset, length);
        return memory;
    }

    /**
     * Allocate a memory block and copy the specified bytes to it.
     *
     * @param bytes the bytes to copy
     * @return the allocated memory block
     */
    public static IAllocatedMemory copyBytes(byte[] bytes) {
        return copyBytes(bytes, 0, bytes.length);
    }

    /**
     * Allocate a memory block and copy the specified string to it.
     *
     * @param string the string to copy
     * @return the allocated memory block
     */
    public static IAllocatedMemory copyString(String string) {
        return copyBytes(string.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Allocate a memory block and copy the specified C string to it.
     * The C string is a null-terminated string.
     *
     * @param string the C string to copy
     * @return the allocated memory block
     */
    public static IAllocatedMemory copyCString(String string) {
        byte[] bytes = string.getBytes(StandardCharsets.UTF_8);
        byte[] cString = new byte[bytes.length + 1];
        System.arraycopy(bytes, 0, cString, 0, bytes.length);
        cString[bytes.length] = 0;
        return copyBytes(cString);
    }

}
