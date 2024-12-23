package com.lyihub.privacy_radar.adapter

import android.content.Context
import android.view.ViewGroup
import android.widget.AdapterView
import androidx.recyclerview.widget.RecyclerView
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.data.SmsInfo
import com.lyihub.privacy_radar.holder.SmsContentHolder


class SmsAdapter(context: Context, listener: AdapterView.OnItemClickListener) :
    BaseRecycleAdapter<SmsInfo, RecyclerView.ViewHolder>(context, listener) {

    override fun onCreateHeadVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder? {
        return null
    }

    override fun onBindHeadVHolder(viewHolder: RecyclerView.ViewHolder, data: SmsInfo?, position: Int) {
    }

    override fun onCreateContentVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return SmsContentHolder(inflate(R.layout.rv_sms_cell, parent))
    }

    override fun onBindContentVHolder(viewHolder: RecyclerView.ViewHolder, data: SmsInfo?, position: Int) {
        val contentViewHolder = viewHolder as SmsContentHolder
        contentViewHolder.mOnItemClickListener = listener
        contentViewHolder.bindData(data)
    }
}