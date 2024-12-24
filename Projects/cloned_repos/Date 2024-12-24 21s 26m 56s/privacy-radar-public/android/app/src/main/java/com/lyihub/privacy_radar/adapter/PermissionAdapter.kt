package com.lyihub.privacy_radar.adapter

import android.content.Context
import android.view.ViewGroup
import android.widget.AdapterView
import androidx.recyclerview.widget.RecyclerView
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.data.PermissionInfo
import com.lyihub.privacy_radar.holder.ContactContentHolder
import com.lyihub.privacy_radar.holder.PermissionContentHolder


class PermissionAdapter(context: Context, listener: AdapterView.OnItemClickListener) :
    BaseRecycleAdapter<PermissionInfo, RecyclerView.ViewHolder>(context, listener) {

    override fun onCreateHeadVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder? {
        return null
    }

    override fun onBindHeadVHolder(viewHolder: RecyclerView.ViewHolder, data: PermissionInfo?, position: Int) {
    }

    override fun onCreateContentVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return PermissionContentHolder(inflate(R.layout.rv_permission_cell, parent))
    }

    override fun onBindContentVHolder(viewHolder: RecyclerView.ViewHolder, data: PermissionInfo?, position: Int) {
        val contentViewHolder = viewHolder as PermissionContentHolder
        contentViewHolder.mOnItemClickListener = listener
        contentViewHolder.bindData(data)
    }
}