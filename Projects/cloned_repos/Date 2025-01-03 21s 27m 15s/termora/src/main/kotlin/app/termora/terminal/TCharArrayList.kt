package app.termora.terminal

import kotlin.math.max

class TCharArrayList {

    private var data = CharArray(80)
    private var pos = 0

    fun add(value: Char) {
        ensureCapacity(pos + 1)
        data[pos++] = value
    }

    fun size(): Int {
        return pos
    }

    fun get(index: Int): Char {
        return data[index]
    }

    fun remove(index: Int): Char {
        val old = data[index]
        remove(index, 1)
        return old
    }

    fun insert(offset: Int, value: Char) {
        if (offset == pos) {
            add(value)
            return
        }
        ensureCapacity(pos + 1)
        System.arraycopy(data, offset, data, offset + 1, pos - offset)
        data[offset] = value
        pos++
    }

    fun remove(offset: Int, length: Int) {
        if (offset < 0 || offset >= pos) {
            throw ArrayIndexOutOfBoundsException(offset)
        }
        if (offset == 0) {
            System.arraycopy(data, length, data, 0, pos - length)
        } else if (pos - length != offset) {
            // data in the middle
            System.arraycopy(
                data, offset + length,
                data, offset, pos - (offset + length)
            )
        }
        pos -= length
    }

    fun isEmpty(): Boolean {
        return pos == 0
    }

    fun set(index: Int, value: Char) {
        data[index] = value
    }

    private fun ensureCapacity(capacity: Int) {
        if (capacity > data.size) {
            val newCap = max((data.size shl 1), capacity)
            val tmp = CharArray(newCap)
            System.arraycopy(data, 0, tmp, 0, data.size)
            data = tmp
        }
    }
}