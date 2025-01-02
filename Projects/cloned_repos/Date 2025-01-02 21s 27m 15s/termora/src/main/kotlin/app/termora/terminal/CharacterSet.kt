package app.termora.terminal

private interface Mapper {
    fun map(ch: Char): Char
}


enum class Graphic {
    G0,
    G1,
    G2,
    G3
}

data class GraphicCharacterSet(
    var graphic: Graphic = Graphic.G0,
    val characterSets: MutableMap<Graphic, CharacterSet> = mutableMapOf(
        Graphic.G0 to CharacterSet.USASCII,
        Graphic.G1 to CharacterSet.USASCII,
        Graphic.G2 to CharacterSet.USASCII,
        Graphic.G3 to CharacterSet.USASCII
    )
) : Mapper {

    private var graphicOnce: Graphic? = null


    fun use(graphic: Graphic) {
        this.graphic = graphic
    }

    /**
     * 只使用一次，对下个调用 [map] 方法的字符生效
     */
    fun useOnce(graphic: Graphic) {
        this.graphicOnce = graphic
    }

    /**
     * 只会指定字符集但不会修改当前的寄存器
     */
    fun designate(graphic: Graphic, characterSet: CharacterSet) {
        characterSets[graphic] = characterSet
    }


    override fun map(ch: Char): Char {
        val char: Char
        if (graphicOnce != null) {
            char = characterSets.getValue(graphicOnce!!).map(ch)
            graphicOnce = null
        } else {
            char = characterSets.getValue(graphic).map(ch)
        }
        return char
    }
}

enum class CharacterSet(private val mapper: Mapper) : Mapper {

    /**
     * United Kingdom (UK), VT100.
     */
    UK(UKMapper()),


    /**
     * United States (USASCII), VT100.
     */
    USASCII(USASCIIMapper()),

    /**
     * DEC Special Character and Line Drawing Set, VT100.
     */
    DECSpecialCharacter(DECSpecialCharacterMapper()),

    /**
     * Dutch, VT200.
     */
    Dutch(DutchMapper()),

    /**
     * Finnish, VT200.
     */
    Finnish(FinnishMapper()),

    /**
     * French, VT200.
     */
    French(FrenchMapper()),

    /**
     * French Canadian, VT200.
     */
    FrenchCanadian(FrenchCanadianMapper()),

    /**
     * German, VT200.
     */
    German(GermanMapper()),

    /**
     * Italian, VT200.
     */
    Italian(ItalianMapper()),

    /**
     * Norwegian/Danish, VT200.
     */
    Norwegian_Danish(Norwegian_DanishMapper()),

    /**
     * Spanish, VT200.
     */
    Spanish(SpanishMapper()),

    /**
     * Swedish, VT200.
     */
    Swedish(SwedishMapper()),

    /**
     * Swiss, VT200.
     */
    Swiss(SwissMapper()),

    /**
     * DEC Supplemental, VT200.
     */
    DECSupplemental(DECSupplementalMapper()),

    ;

    override fun map(ch: Char): Char {
        return mapper.map(ch)
    }
}


/**
 * https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h4-Controls-beginning-with-ESC:ESC-lparen-C:C-=-lt.10A2
 */
private class DECSupplementalMapper : Mapper {
    override fun map(ch: Char): Char {
        if (ch.code in 0..63) {
            return (ch.code + 160).toChar()
        }
        return ch
    }
}

/**
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L207
 */
private class SwissMapper : Mapper {
    private val mapping = mapOf(
        '#' to 'ù',
        '@' to 'à',
        '[' to 'é',
        '\\' to 'ç',
        ']' to 'ê',
        '^' to 'î',
        '_' to 'è',
        '`' to 'ô',
        '{' to 'ä',
        '|' to 'ö',
        '}' to 'ü',
        '~' to 'û',
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}


/**
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L207
 */
private class SwedishMapper : Mapper {
    private val mapping = mapOf(
        '@' to 'É',
        '[' to 'Ä',
        '\\' to 'Ö',
        ']' to 'Å',
        '^' to 'Ü',
        '`' to 'é',
        '{' to 'ä',
        '|' to 'ö',
        '}' to 'å',
        '~' to 'ü',
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}

/**
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L207
 */
private class SpanishMapper : Mapper {
    private val mapping = mapOf(
        '#' to '£',
        '@' to '§',
        '[' to '¡',
        '\\' to 'Ñ',
        ']' to '¿',
        '{' to '°',
        '|' to 'ñ',
        '}' to 'ç',
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}


/**
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L189
 */
private class Norwegian_DanishMapper : Mapper {
    private val mapping = mapOf(
        '@' to 'Ä',
        '[' to 'Æ',
        '\\' to 'Ø',
        ']' to 'Å',
        '^' to 'Ü',
        '`' to 'ä',
        '{' to 'æ',
        '|' to 'ø',
        '}' to 'å',
        '~' to 'ü',
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}


/**
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L170
 */
private class ItalianMapper : Mapper {
    private val mapping = mapOf(
        '#' to '£',
        '@' to '§',
        '[' to '°',
        '\\' to 'ç',
        ']' to 'é',
        '`' to 'ù',
        '{' to 'à',
        '|' to 'ò',
        '}' to 'è',
        '~' to 'ì',
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}


/**
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L154
 */
private class GermanMapper : Mapper {
    private val mapping = mapOf(
        '@' to '§',
        '[' to 'Ä',
        '\\' to 'Ö',
        ']' to 'Ü',
        '{' to 'ä',
        '|' to 'ö',
        '}' to 'ü',
        '~' to 'ß',
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}


/**
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L136
 */
private class FrenchCanadianMapper : Mapper {
    private val mapping = mapOf(
        '@' to 'à',
        '[' to 'â',
        '\\' to 'ç',
        ']' to 'ê',
        '^' to 'î',
        '`' to 'ô',
        '{' to 'é',
        '|' to 'ù',
        '}' to 'è',
        '~' to 'û',
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}


/**
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L119
 */
private class FrenchMapper : Mapper {
    private val mapping = mapOf(
        '#' to '£',
        '@' to 'à',
        '[' to '°',
        '\\' to 'ç',
        ']' to '§',
        '{' to 'é',
        '|' to 'ù',
        '}' to 'è',
        '~' to '¨'
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}

/**
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L101
 */
private class FinnishMapper : Mapper {
    private val mapping = mapOf(
        '[' to 'Ä',
        '\\' to 'Ö',
        ']' to 'Å',
        '^' to 'Ü',
        '`' to 'é',
        '{' to 'ä',
        '|' to 'ö',
        '}' to 'å',
        '~' to 'ü',
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}

/**
 * https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h4-Controls-beginning-with-ESC:ESC-lparen-C:C-=-4.FF6
 * https://github.com/xtermjs/xterm.js/blob/5.5.0/src/common/data/Charsets.ts#L84
 */
private class DutchMapper : Mapper {
    private val mapping = mapOf(
        '#' to '£',
        '@' to '¾',
        '[' to 'ĳ',
        '\\' to '½',
        ']' to '|',
        '{' to '¨',
        '|' to 'f',
        '}' to '¼',
        '~' to '´',
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}

/**
 * https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h4-Controls-beginning-with-ESC:ESC-lparen-C:C-=-A.1003
 */
private class UKMapper : Mapper {
    override fun map(ch: Char): Char {
        if (ch.code == 3) {
            return '\u00a3'
        }
        return ch
    }
}

/**
 * https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h4-Controls-beginning-with-ESC:ESC-lparen-C:C-=-B.1004
 */
private class USASCIIMapper : Mapper {
    override fun map(ch: Char): Char {
        return ch
    }
}

/**
 * https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h4-Controls-beginning-with-ESC:ESC-lparen-C:C-=-0.FF2
 */
private class DECSpecialCharacterMapper : Mapper {
    private val mapping = mapOf(
        '`' to '\u25c6', // '◆'
        'a' to '\u2592', // '▒'
        'b' to '\u2409', // '␉' (HT)
        'c' to '\u240c', // '␌' (FF)
        'd' to '\u240d', // '␍' (CR)
        'e' to '\u240a', // '␊' (LF)
        'f' to '\u00b0', // '°'
        'g' to '\u00b1', // '±'
        'h' to '\u2424', // '␤' (NL)
        'i' to '\u240b', // '␋' (VT)
        'j' to '\u2518', // '┘'
        'k' to '\u2510', // '┐'
        'l' to '\u250c', // '┌'
        'm' to '\u2514', // '└'
        'n' to '\u253c', // '┼'
        'o' to '\u23ba', // '⎺'
        'p' to '\u23bb', // '⎻'
        'q' to '\u2500', // '─'
        'r' to '\u23bc', // '⎼'
        's' to '\u23bd', // '⎽'
        't' to '\u251c', // '├'
        'u' to '\u2524', // '┤'
        'v' to '\u2534', // '┴'
        'w' to '\u252c', // '┬'
        'x' to '\u2502', // '│'
        'y' to '\u2264', // '≤'
        'z' to '\u2265', // '≥'
        '{' to '\u03c0', // 'π'
        '|' to '\u2260', // '≠'
        '}' to '\u00a3', // '£'
        '~' to '\u00b7'  // '·'
    )

    override fun map(ch: Char): Char {
        return mapping.getOrDefault(ch, ch)
    }
}