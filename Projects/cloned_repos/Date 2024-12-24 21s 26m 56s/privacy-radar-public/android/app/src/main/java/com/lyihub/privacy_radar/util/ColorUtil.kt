package com.lyihub.privacy_radar.util

import androidx.annotation.ColorInt
import com.lyihub.privacy_radar.R
import java.util.*


object ColorUtil {
    fun getDefaultRandomColor(): Int {
//        val colors = intArrayOf(
//            R.color.color_1AEB4F3A,
//            R.color.color_1CEB4F3A,
//            R.color.color_1FEB4F3A,
//            R.color.color_21EB4F3A)

        val colors = intArrayOf(
            R.color.transparent)

        val random = Random()

        return colors[random.nextInt(colors.size)]
//        var color = Color.argb(255,
//            random.nextInt(256),
//            random.nextInt(256),
//            random.nextInt(256))

//        return color
    }

    /**
     * Set the alpha component of `color` to be `alpha`.
     */
    fun modifyAlpha(@ColorInt color: Int, alpha: Int): Int {
        return color and 0x00ffffff or (alpha shl 24)
    }

    /**
     * Set the alpha component of `color` to be `alpha`.
     */
    fun modifyAlpha(@ColorInt color: Int, alpha: Float): Int {
        return modifyAlpha(color, (255f * alpha).toInt())
    }

    /**
     * 颜色透明度16进制计算
     * 255 * 不透明度 -> 转换成16进制数
     * @param alpha 不透明度
     */
    fun calcuColorAlpha (alpha: Int): String {
        var temp = 255 * alpha * 1.0f / 100f
        var alpha = Math.round(temp)
        var hexStr = Integer.toHexString(alpha)
        if (hexStr.length < 2) {
            hexStr = "0$hexStr"
        }
        return "$alpha% = ${hexStr.toUpperCase()}"
    }
}