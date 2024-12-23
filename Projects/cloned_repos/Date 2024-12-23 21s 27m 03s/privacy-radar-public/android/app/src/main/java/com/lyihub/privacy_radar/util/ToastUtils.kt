package com.lyihub.privacy_radar.util

import android.content.Context
import android.text.TextUtils
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.widget.TextView
import android.widget.Toast
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.app.App


object ToastUtils {

    /**
     * 短暂显示
     *
     * @param msg
     */
    fun showShort(msg: CharSequence) {
        Toast.makeText(App.get(), msg, Toast.LENGTH_SHORT).show()
    }

    /**
     * 短暂显示
     *
     * @param resId
     */
    fun showShort(resId: Int) {
        val text = ResUtils.getStringRes(resId)
        Toast.makeText(App.get(), text, Toast.LENGTH_SHORT).show()
    }

    /**
     * 长时间显示
     *
     * @param msg
     */
    fun showLong(msg: CharSequence) {
        Toast.makeText(App.get(), msg, Toast.LENGTH_LONG).show()
    }

    /**
     * 短暂显示
     *
     * @param resId
     */
    fun showLong(resId: Int) {
        val text = ResUtils.getStringRes(resId)
        Toast.makeText(App.get(), text, Toast.LENGTH_LONG).show()
    }

    fun show(msg: CharSequence?) {
        if (msg == null) return
        if (TextUtils.isEmpty(msg.toString())) return

        val inflater = App.get().getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        //自定义布局
        val view: View = inflater.inflate(R.layout.toast_layout, null)
        val mTvMessage = view.findViewById<View>(R.id.tv_message) as TextView
        mTvMessage.text = msg
        val mToast = Toast(App.get())
        val height = ScreenUtils.getHeight(App.get())
        mToast.setGravity(Gravity.BOTTOM, 0, height / 6)
        mToast.duration = Toast.LENGTH_SHORT
        mToast.view = view
        mToast.show()
    }

    fun show(resId: Int) {
        val msg = ResUtils.getStringRes(resId)
        if (TextUtils.isEmpty(msg)) return
        val inflater = App.get().getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        //自定义布局
        val view: View = inflater.inflate(R.layout.toast_layout, null)
        val mTvMessage = view.findViewById<View>(R.id.tv_message) as TextView
        mTvMessage.text = msg
        val mToast = Toast(App.get())
        val height = ScreenUtils.getHeight(App.get())
        mToast.setGravity(Gravity.BOTTOM, 0, height / 6)
        mToast.duration = Toast.LENGTH_SHORT
        mToast.view = view
        mToast.show()
    }

}