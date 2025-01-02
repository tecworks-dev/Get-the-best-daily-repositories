package zmodem

import org.apache.commons.net.io.CopyStreamEvent

/**
 * 如果一共两个文件，并且传输第一个文件时：
 *
 * remaining = 1
 * index = 1
 */
class FileCopyStreamEvent(
    source: Any,
    // 本次传输的文件名
    val filename: String,
    // 剩余未传输的文件数量
    val remaining: Int,
    // 第几个文件
    val index: Int,
    // 总字节数
    totalBytesTransferred: Long,
    // 已经传输完成的字节数
    bytesTransferred: Int,
    // 本次传输的字节数
    streamSize: Long,
    /**
     * 这个文件被跳过了
     */
    val skip: Boolean = false,
) :
    CopyStreamEvent(
        source, totalBytesTransferred,
        bytesTransferred,
        streamSize
    )