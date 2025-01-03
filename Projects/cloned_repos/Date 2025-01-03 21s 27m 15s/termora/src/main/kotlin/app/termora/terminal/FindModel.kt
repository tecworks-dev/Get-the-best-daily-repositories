package app.termora.terminal


/**
 * 当搜索完毕后，搜索结果会被缓存
 */
interface FindModel {
    /**
     * 获取终端
     */
    fun getTerminal(): Terminal

    /**
     * 搜索
     */
    fun find(text: String, ignoreCase: Boolean = true): List<FindKind>

}