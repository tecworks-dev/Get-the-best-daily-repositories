package io.github.duzhaokun123.yapatch.patch

object Versions {
    val yapatch = "0.1.0"
    val pine = "0.3.0"
    val pineXposed = "0.2.0"

    override fun toString(): String {
        return """
            yapatch: $yapatch
            pine: $pine
            pineXposed: $pineXposed
        """.trimIndent()
    }
}