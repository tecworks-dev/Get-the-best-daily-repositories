package com.lyihub.privacy_radar.holder

import android.view.View
import com.lyihub.privacy_radar.data.ContactInfo
import kotlinx.android.synthetic.main.rv_contact_cell.view.*


class ContactContentHolder(itemView: View) : ContentViewHolder(itemView) {

    fun bindData(data: ContactInfo?) {
        itemView.mTvName.text = data?.name ?: ""
        itemView.mTvPhone.text = data?.phone ?: ""
    }

    override fun onLongClick(v: View): Boolean {
        return false
    }
}