package app.termora.findeverywhere

interface FindEverywhereProvider {

    /**
     * 搜索
     */
    fun find(pattern: String): List<FindEverywhereResult>

    /**
     * 如果返回非空，表示单独分组
     */
    fun group(): String = "Default Group"

    /**
     * 越小越靠前
     */
    fun order(): Int = Integer.MAX_VALUE
}