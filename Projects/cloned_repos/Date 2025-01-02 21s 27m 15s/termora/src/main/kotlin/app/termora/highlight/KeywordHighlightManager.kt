package app.termora.highlight

import app.termora.TerminalPanelFactory
import app.termora.db.Database
import org.slf4j.LoggerFactory

class KeywordHighlightManager private constructor() {

    companion object {
        val instance by lazy { KeywordHighlightManager() }
        private val log = LoggerFactory.getLogger(KeywordHighlightManager::class.java)
    }

    private val database by lazy { Database.instance }
    private val keywordHighlights = mutableMapOf<String, KeywordHighlight>()

    init {
        keywordHighlights.putAll(database.getKeywordHighlights().associateBy { it.id })
    }


    fun addKeywordHighlight(keywordHighlight: KeywordHighlight) {
        database.addKeywordHighlight(keywordHighlight)
        keywordHighlights[keywordHighlight.id] = keywordHighlight
        TerminalPanelFactory.instance.repaintAll()

        if (log.isDebugEnabled) {
            log.debug("Keyword highlighter added. {}", keywordHighlight)
        }
    }

    fun removeKeywordHighlight(id: String) {
        database.removeKeywordHighlight(id)
        keywordHighlights.remove(id)
        TerminalPanelFactory.instance.repaintAll()

        if (log.isDebugEnabled) {
            log.debug("Keyword highlighter removed. {}", id)
        }
    }

    fun getKeywordHighlights(): List<KeywordHighlight> {
        return keywordHighlights.values.sortedBy { it.sort }
    }

    fun getKeywordHighlight(id: String): KeywordHighlight? {
        return keywordHighlights[id]
    }
}