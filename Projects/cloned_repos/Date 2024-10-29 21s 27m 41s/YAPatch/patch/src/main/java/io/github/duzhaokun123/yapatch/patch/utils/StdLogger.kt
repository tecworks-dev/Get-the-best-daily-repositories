package io.github.duzhaokun123.yapatch.patch.utils

object StdLogger: Logger {
    override fun info(message: String) {
        println(message)
    }

    override fun warn(message: String) {
        println(message)
    }

    override fun error(message: String) {
        println(message)
    }
}