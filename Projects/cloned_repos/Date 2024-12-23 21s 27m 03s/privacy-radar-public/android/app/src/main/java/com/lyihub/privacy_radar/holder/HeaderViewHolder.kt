package com.lyihub.privacy_radar.holder

import android.view.View
import android.widget.AdapterView
import androidx.recyclerview.widget.RecyclerView


open class HeaderViewHolder: RecyclerView.ViewHolder,View.OnClickListener {
    var mOnItemClickListener: AdapterView.OnItemClickListener? = null

    constructor(itemView: View) : super(itemView) {
        itemView.setOnClickListener(this)
    }

    override fun onClick(v: View?) {
        mOnItemClickListener?.onItemClick(null, v, bindingAdapterPosition, 0)
    }
}