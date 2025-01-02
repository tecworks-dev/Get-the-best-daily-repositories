package app.termora.terminal

import java.nio.ByteBuffer


interface PtyConnector {

    /**
     * 读取
     */
    fun read(buffer: CharArray): Int

    /**
     * 将数据写入
     */
    fun write(buffer: ByteArray, offset: Int, len: Int)

    fun write(buffer: ByteArray) {
        write(buffer, 0, buffer.size)
    }

    fun write(buffer: String) {
        if (buffer.isEmpty()) return
        write(buffer.toByteArray())
    }

    fun write(buffer: Int) {
        write(ByteBuffer.allocate(Integer.BYTES).putInt(buffer).flip().array())
    }

    /**
     * 修改 pty 大小
     */
    fun resize(rows: Int, cols: Int)

    /**
     * 等待断开
     */
    fun waitFor(): Int

    /**
     * 关闭
     */
    fun close()

}