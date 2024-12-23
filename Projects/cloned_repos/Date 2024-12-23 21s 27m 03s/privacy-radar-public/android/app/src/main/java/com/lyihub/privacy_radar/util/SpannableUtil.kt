package com.lyihub.privacy_radar.util

import android.graphics.Typeface
import android.text.*
import android.text.style.*
import android.widget.TextView



object SpannableUtil {

    val TAG = "SpannableUtil"

    /**
     * 设置不同颜色
     */
    fun setSpannableColor(textView: TextView?,color: Int,text: String?,spanText: String?) {
        if (TextUtils.isEmpty(text)) return
        if (TextUtils.isEmpty(spanText)) {
            textView?.text = text
            return
        }
        try {
            val mSearchCount = getWordCount(spanText)
            val spannableString = SpannableString(text)
            var index = 0
            while (index != -1) {
                index = text?.indexOf(spanText ?: "", index) ?: 0
                if (index == -1) break
                spannableString.setSpan(
                    ForegroundColorSpan(color),
                    index,
                    (index + mSearchCount).also { index = it },
                    Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                )
            }
            textView?.text = spannableString
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun getSpannableTextSize(textSize: Int,text: String?,spanText: String?): SpannableString? {
        if (TextUtils.isEmpty(text)) return null
        val spannableString = SpannableString(text)
        try {
            val mSearchCount = getWordCount(spanText)
            var index = 0
            while (index != -1) {
                index = text?.indexOf(spanText ?: "", index) ?: 0
                if (index == -1) break
                spannableString.setSpan(
                    AbsoluteSizeSpan(textSize),
                    index,
                    (index + mSearchCount).also { index = it },
                    Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return spannableString
    }

    fun getBoldSizeSpannable(text: String?,spanTexts: List<String>?,textSize: Int): SpannableString? {
        if (spanTexts == null) return null
        if (spanTexts.isEmpty()) return null

        try {

        } catch (e: Exception) {
            e.printStackTrace()
        }

        val spannableString = SpannableString(text)

        try {
            spanTexts.forEach {spanText ->
                val mSearchCount = getWordCount(spanText)

                var index = 0
                while (index != -1) {
                    index = text?.indexOf(spanText, index) ?: 0
                    if (index == -1) break
                    spannableString.setSpan(
                        AbsoluteSizeSpan(textSize),
                        index,
                        (index + mSearchCount),
                        Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                    )
                    spannableString.setSpan(
                        StyleSpan(Typeface.BOLD),
                        index,
                        (index + mSearchCount),
                        Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                    )
                    index += mSearchCount
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        return spannableString
    }

    fun getBoldColorSizeSpannable(text: String?,spanTexts: List<String>?,textSize: Int,color: Int): SpannableString? {
        if (spanTexts == null) return null
        if (spanTexts.size == 0) return null

        val spannableString = SpannableString(text)

        try {
            spanTexts.forEach {spanText ->
                val mSearchCount = getWordCount(spanText)

                var index = 0
                while (index != -1) {
                    index = text?.indexOf(spanText, index) ?: 0
                    if (index == -1) break
                    spannableString.setSpan(
                        ForegroundColorSpan(color),
                        index,
                        (index + mSearchCount),
                        Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                    )
                    spannableString.setSpan(
                        AbsoluteSizeSpan(textSize),
                        index,
                        (index + mSearchCount),
                        Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                    )
                    spannableString.setSpan(
                        StyleSpan(Typeface.BOLD),
                        index,
                        (index + mSearchCount),
                        Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                    )
                    index += mSearchCount
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        return spannableString
    }

    fun getSpannableBoldText(text: String?,spanText: String?): SpannableString? {
        if (TextUtils.isEmpty(text)) return null
        val spannableString = SpannableString(text)
        try {
            val mSearchCount = getWordCount(spanText)
            var index = 0
            while (index != -1) {
                index = text?.indexOf(spanText ?: "", index) ?: 0
                if (index == -1) break
                spannableString.setSpan(
                    StyleSpan(Typeface.BOLD),
                    index,
                    (index + mSearchCount).also { index = it },
                    Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return spannableString
    }

    fun getSpannableBoldText(text: SpannableString?,spanText: String?): SpannableString? {
        if (text == null) return null
        if (TextUtils.isEmpty(text)) return null
        val spannableString = SpannableString(text)
        try {
            val mSearchCount = getWordCount(spanText)
            var index = 0
            while (index != -1) {
                index = text?.indexOf(spanText ?: "", index) ?: 0
                if (index == -1) break
                spannableString.setSpan(
                    StyleSpan(Typeface.BOLD),
                    index,
                    (index + mSearchCount).also { index = it },
                    Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return spannableString
    }

    fun getSpannableColorText(text: SpannableString?,spanText: String?,color: Int): SpannableString? {
        if (text == null) return null
        if (TextUtils.isEmpty(text)) return null
        val spannableString = SpannableString(text)
        try {
            val mSearchCount = getWordCount(spanText)
            var index = 0
            while (index != -1) {
                index = text?.indexOf(spanText ?: "", index) ?: 0
                if (index == -1) break
                spannableString.setSpan(
                    ForegroundColorSpan(color),
                    index,
                    (index + mSearchCount).also { index = it },
                    Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return spannableString
    }

    fun getSpannableTextSize(textSize: Int,text: String?,spanTexts: List<String>?): SpannableString? {
        if (TextUtils.isEmpty(text)) return null

        val spannableString = SpannableString(text)

        try {
            spanTexts?.forEach {
                var spanText = it
                val redCount = getWordCount(spanText)

                var index = 0
                while (index != -1) {
                    index = text?.indexOf(spanText, index) ?: 0
                    if (index == -1) break
                    spannableString.setSpan(
                        AbsoluteSizeSpan(textSize),
                        index,
                        (index + redCount).also { index = it },
                        Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                    )
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        return spannableString
    }

    fun setSpannableColor(textView: TextView?,color: Int,text: String?,spanTexts: List<String>?) {
        val spannableString = SpannableString(text)

        try {
            var size = spanTexts?.size ?: 0
            for (i in 0 until size) {
                var spanText = spanTexts?.get(i) ?: ""
                val redCount = getWordCount(spanText)

                var index = 0
                while (index != -1) {
                    index = text?.indexOf(spanText, index) ?: 0
                    if (index == -1) break
                    spannableString.setSpan(
                        ForegroundColorSpan(color),
                        index,
                        (index + redCount).also { index = it },
                        Spanned.SPAN_INCLUSIVE_EXCLUSIVE
                    )
                }
            }

            textView?.text = spannableString
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * 添加下划线
     */
    fun setSpannableUnderline(textView: TextView?,text: String?,spanText: String?) {
        if (textView == null) return
        if (TextUtils.isEmpty(text)) return
        if (TextUtils.isEmpty(spanText)) return

        try {
            val mUnderlineCount = getWordCount(spanText)
            val spannableString = SpannableString(text)

            var underlineIndex = 0
            while (underlineIndex != -1) {
                underlineIndex = text?.indexOf(spanText ?: "", underlineIndex) ?: 0
                if (underlineIndex == -1) break
                spannableString.setSpan(
                    UnderlineSpan(),
                    underlineIndex,
                    (underlineIndex + mUnderlineCount).also { underlineIndex = it },
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }

            textView.text = spannableString
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * 添加删除线
     */
    fun setSpannableDeleteline(textView: TextView?,text: String?,spanText: String?) {
        if (textView == null) return
        if (TextUtils.isEmpty(text)) return
        if (TextUtils.isEmpty(spanText)) return

        try {
            val mUnderlineCount = getWordCount(spanText)
            val spannableString = SpannableString(text)

            var underlineIndex = 0
            while (underlineIndex != -1) {
                underlineIndex = text?.indexOf(spanText ?: "", underlineIndex) ?: 0
                if (underlineIndex == -1) break
                spannableString.setSpan(
                    StrikethroughSpan(),
                    underlineIndex,
                    (underlineIndex + mUnderlineCount).also { underlineIndex = it },
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }

            textView.text = spannableString
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }


    fun getWordCount(s: String?): Int {
        try {
            var s = s
            s = s?.replace("[\\u4e00-\\u9fa5]".toRegex(), "*")
            return s?.length ?: 0
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return 0
    }

}