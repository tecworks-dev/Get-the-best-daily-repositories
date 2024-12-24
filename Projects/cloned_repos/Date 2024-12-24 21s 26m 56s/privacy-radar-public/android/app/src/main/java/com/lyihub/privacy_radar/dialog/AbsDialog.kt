package com.lyihub.privacy_radar.dialog

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.view.Window
import android.view.WindowManager
import com.lyihub.privacy_radar.R


abstract class AbsDialog(context: Context): Dialog(context, R.style.BaseNoTitleDialog) {

    protected abstract fun bindContentView(): Int

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(bindContentView())

        //设置属性信息宽高或者动画
        val window = window
        handleWindow(window!!)
        val wlp = window.attributes
        handleLayoutParams(wlp)
        window.attributes = wlp
        //禁止app录屏和截屏
//        window.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
    }

    /**
     * 用于处理窗口的属性
     *
     * @param window
     */
    abstract fun handleWindow(window: Window)

    abstract fun handleLayoutParams(wlp: WindowManager.LayoutParams?)

}