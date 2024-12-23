package com.lyihub.privacy_radar.dialog

import android.content.Context
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.Window
import android.view.WindowManager
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.interfaces.OnDialogOkCancelClickListener
import com.lyihub.privacy_radar.util.ScreenUtils
import kotlinx.android.synthetic.main.dlg_common_tip.*


class CommonTipDialog(context: Context): AbsDialog(context),View.OnClickListener {

    var mTitle: String? = null
    var mCancelText: String? = null
    var mOkText: String? = null

    var okBtnVisible: Int = View.VISIBLE
    var cancelBtnVisible: Int = View.VISIBLE

    var mOnDialogOkCancelClickListener: OnDialogOkCancelClickListener? = null

    override fun bindContentView() = R.layout.dlg_common_tip

    override fun handleWindow(window: Window) {
        window.setGravity(Gravity.CENTER)
    }

    override fun handleLayoutParams(wlp: WindowManager.LayoutParams?) {
        wlp?.width = (ScreenUtils.getWidth(context) * 0.8).toInt()
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initialize()
    }

    fun initialize () {
        mTvTitle.text = mTitle
        mTvCancel.text = mCancelText
        mTvOk.text = mOkText

        mTvOk.visibility = okBtnVisible
        mTvCancel.visibility = cancelBtnVisible

        mTvOk.setOnClickListener(this)
        mTvCancel.setOnClickListener(this)
    }

    override fun onClick(v: View?) {
        when (v) {
            mTvOk -> {
                mOnDialogOkCancelClickListener?.OnDialogOkClick()
                dismiss()
            }
            mTvCancel -> {
                mOnDialogOkCancelClickListener?.OnDialogCancelClick()
                dismiss()
            }
        }
    }

}