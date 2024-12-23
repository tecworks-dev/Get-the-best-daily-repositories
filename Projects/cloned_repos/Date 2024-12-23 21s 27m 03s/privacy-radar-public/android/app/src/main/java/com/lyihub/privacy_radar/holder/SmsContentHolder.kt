package com.lyihub.privacy_radar.holder

import android.view.View
import com.lyihub.privacy_radar.data.SmsInfo
import kotlinx.android.synthetic.main.rv_sms_cell.view.*


class SmsContentHolder(itemView: View) : ContentViewHolder(itemView) {

    fun bindData(data: SmsInfo?) {
        itemView.mTvAddress.text = data?.address ?: ""
        itemView.mTvBody.text = data?.body ?: ""
    }

    override fun onLongClick(v: View): Boolean {
        return false
    }
}