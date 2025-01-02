package app.termora.terminal

import kotlin.math.max

class TTextStyleArrayList {

    private var data = LongArray(80)
    private var pos = 0

    fun add(value: Long) {
        ensureCapacity(pos + 1)
        data[pos++] = value
    }

    fun add(value: TextStyle) {
        add(value.value)
    }

    fun size(): Int {
        return pos
    }

    fun get(index: Int): Long {
        return data[index]
    }

    fun remove(index: Int): Long {
        val old = data[index]
        remove(index, 1)
        return old
    }

    fun insert(offset: Int, value: Long) {
        if (offset == pos) {
            add(value)
            return
        }
        ensureCapacity(pos + 1)
        System.arraycopy(data, offset, data, offset + 1, pos - offset)
        data[offset] = value
        pos++
    }

    fun insert(offset: Int, value: TextStyle) {
        insert(offset, value.value)
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

    fun set(index: Int, value: Long) {
        data[index] = value
    }

    fun set(index: Int, value: TextStyle) {
        set(index, value.value)
    }

    private fun ensureCapacity(capacity: Int) {
        if (capacity > data.size) {
            val newCap = max((data.size shl 1), capacity)
            val tmp = LongArray(newCap)
            System.arraycopy(data, 0, tmp, 0, data.size)
            data = tmp
        }
    }

    fun getTextStyle(index: Int): TextStyle {
        return TextStyle(get(index))
    }
}