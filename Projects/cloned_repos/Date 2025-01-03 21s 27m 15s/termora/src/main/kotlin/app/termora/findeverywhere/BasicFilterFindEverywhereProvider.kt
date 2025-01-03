package app.termora.findeverywhere

class BasicFilterFindEverywhereProvider(private val provider: FindEverywhereProvider) : FindEverywhereProvider {
    override fun find(pattern: String): List<FindEverywhereResult> {
        val results = provider.find(pattern)
        if (pattern.isBlank()) {
            return results
        }
        return results.filter {
            it.toString().contains(pattern, true)
        }
    }

    override fun order(): Int {
        return provider.order()
    }


    override fun group(): String {
        return provider.group()
    }
}