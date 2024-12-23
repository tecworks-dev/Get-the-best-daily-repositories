package com.lyihub.privacy_radar.holder

import android.view.View
import com.lyihub.privacy_radar.data.CallLogInfo
import kotlinx.android.synthetic.main.rv_call_log_cell.view.*


class CallLogContentHolder(itemView: View) : ContentViewHolder(itemView) {

    fun bindData(data: CallLogInfo?) {
        itemView.mTvName.text = data?.name ?: ""
        itemView.mTvPhone.text = data?.phone ?: ""
    }

    override fun onLongClick(v: View): Boolean {
        return false
    }
}