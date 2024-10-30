package dev.tmpfs.libcoresyscall.core;

import java.io.Closeable;

public interface IAllocatedMemory extends Closeable {

    /**
     * Get the address of the memory block.
     * <p>
     * If the memory block is already freed, it returns 0.
     *
     * @return the address of the memory block.
     */
    long getAddress();

    /**
     * Get the size of the memory block.
     * <p>
     * If the memory block is already freed, it returns 0.
     *
     * @return the size of the memory block in bytes.
     */
    long getSize();

    /**
     * Free the memory block. It is safe to call this method multiple times.
     * After the memory block is freed, calling {@link #getAddress()} and {@link #getSize()} will return 0.
     */
    void free();

    /**
     * Same as {@link #free()}.
     */
    @Override
    void close();

    /**
     * Check if the memory block is already freed.
     */
    boolean isFreed();

}
