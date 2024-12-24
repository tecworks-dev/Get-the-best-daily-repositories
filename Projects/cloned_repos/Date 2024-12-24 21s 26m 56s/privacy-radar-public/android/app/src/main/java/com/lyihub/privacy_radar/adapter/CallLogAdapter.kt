package com.lyihub.privacy_radar.adapter

import android.content.Context
import android.view.ViewGroup
import android.widget.AdapterView
import androidx.recyclerview.widget.RecyclerView
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.data.CallLogInfo
import com.lyihub.privacy_radar.holder.CallLogContentHolder

class CallLogAdapter(context: Context, listener: AdapterView.OnItemClickListener) :
    BaseRecycleAdapter<CallLogInfo, RecyclerView.ViewHolder>(context, listener) {

    override fun onCreateHeadVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder? {
        return null
    }

    override fun onBindHeadVHolder(viewHolder: RecyclerView.ViewHolder, data: CallLogInfo?, position: Int) {
    }

    override fun onCreateContentVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return CallLogContentHolder(inflate(R.layout.rv_call_log_cell, parent))
    }

    override fun onBindContentVHolder(viewHolder: RecyclerView.ViewHolder, data: CallLogInfo?, position: Int) {
        val contentViewHolder = viewHolder as CallLogContentHolder
        contentViewHolder.mOnItemClickListener = listener
        contentViewHolder.bindData(data)
    }
}