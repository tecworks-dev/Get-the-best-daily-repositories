package zmodem.util

import java.io.InputStream
import java.io.OutputStream

class EmptyFileAdapter private constructor() : FileAdapter {
    companion object {
        val instance by lazy { EmptyFileAdapter() }
    }

    override fun getName(): String {
        TODO("Not yet implemented")
    }

    override fun getInputStream(): InputStream {
        TODO("Not yet implemented")
    }

    override fun getOutputStream(): OutputStream {
        TODO("Not yet implemented")
    }

    override fun getOutputStream(append: Boolean): OutputStream {
        TODO("Not yet implemented")
    }

    override fun getChild(name: String?): FileAdapter {
        TODO("Not yet implemented")
    }

    override fun length(): Long {
        TODO("Not yet implemented")
    }

    override fun isDirectory(): Boolean {
        return true
    }

    override fun exists(): Boolean {
        return true
    }
}