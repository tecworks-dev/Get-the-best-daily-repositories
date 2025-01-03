package app.termora.terminal

/**
 * example: echo -e "\e[1;5;3;9;4;91mHello World\e[0m"
 */
@JvmInline
value class TextStyle(val value: Long) {

    companion object {
        val Default = TextStyle()
    }

    constructor() : this(0)

    // 前景
    val foreground: Int get() = (value shr 40).toInt() and 0xFFFFFF

    // 背景
    val background: Int get() = (value shr 16).toInt() and 0xFFFFFF

    // 加粗
    val bold: Boolean get() = (value shr 0 and 1L) == 1L

    // 斜体
    val italic: Boolean get() = (value shr 1 and 1L) == 1L

    // 下划线
    val underline: Boolean get() = (value shr 2 and 1L) == 1L

    // 颜色反转
    val inverse: Boolean get() = (value shr 3 and 1L) == 1L

    // 删除线
    val lineThrough: Boolean get() = (value shr 4 and 1L) == 1L

    // 闪烁
    val blink: Boolean get() = (value shr 5 and 1L) == 1L

    // 昏暗
    val dim: Boolean get() = (value shr 6 and 1L) == 1L

    // 双下划线
    val doublyUnderline: Boolean get() = (value shr 7 and 1L) == 1L

    fun foreground(foreground: Int): TextStyle {
        return copy(foreground = foreground)
    }

    fun background(background: Int): TextStyle {
        return copy(background = background)
    }

    fun bold(bold: Boolean): TextStyle {
        return copy(bold = bold)
    }

    fun italic(italic: Boolean): TextStyle {
        return copy(italic = italic)
    }

    fun underline(underline: Boolean): TextStyle {
        return copy(underline = underline)
    }

    fun inverse(inverse: Boolean): TextStyle {
        return copy(inverse = inverse)
    }

    fun lineThrough(lineThrough: Boolean): TextStyle {
        return copy(lineThrough = lineThrough)
    }

    fun blink(blink: Boolean): TextStyle {
        return copy(blink = blink)
    }

    fun dim(dim: Boolean): TextStyle {
        return copy(dim = dim)
    }

    fun doublyUnderline(doublyUnderline: Boolean): TextStyle {
        return copy(doublyUnderline = doublyUnderline)
    }

    fun copyOnlyColors(): TextStyle {
        return TextStyle().copy(foreground = foreground, background = background)
    }

    fun copy(
        foreground: Int = this.foreground,
        background: Int = this.background,
        bold: Boolean = this.bold,
        italic: Boolean = this.italic,
        underline: Boolean = this.underline,
        lineThrough: Boolean = this.lineThrough,
        inverse: Boolean = this.inverse,
        blink: Boolean = this.blink,
        dim: Boolean = this.dim,
        doublyUnderline: Boolean = this.doublyUnderline
    ): TextStyle {
        var value: Long = 0

        value = value or (foreground.toLong() shl 40)
        value = value or (background.toLong() shl 16)

        if (bold)
            value = value or (1L shl 0)
        if (italic)
            value = value or (1L shl 1)
        if (underline)
            value = value or (1L shl 2)
        if (inverse)
            value = value or (1L shl 3)
        if (lineThrough)
            value = value or (1L shl 4)
        if (blink)
            value = value or (1L shl 5)
        if (dim)
            value = value or (1L shl 6)
        if (doublyUnderline)
            value = value or (1L shl 7)

        return TextStyle(value)
    }
}
