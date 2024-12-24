package com.lyihub.privacy_radar.adapter

import android.content.Context
import android.content.pm.ApplicationInfo
import android.view.ViewGroup
import android.widget.AdapterView
import androidx.recyclerview.widget.RecyclerView
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.data.AppInfo
import com.lyihub.privacy_radar.data.ContactInfo
import com.lyihub.privacy_radar.holder.AppContentHolder
import com.lyihub.privacy_radar.holder.ContactContentHolder

class AppAdapter(context: Context, listener: AdapterView.OnItemClickListener) :
    BaseRecycleAdapter<AppInfo, RecyclerView.ViewHolder>(context, listener) {

    override fun onCreateHeadVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder? {
        return null
    }

    override fun onBindHeadVHolder(viewHolder: RecyclerView.ViewHolder, data: AppInfo?, position: Int) {
    }

    override fun onCreateContentVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return AppContentHolder(inflate(R.layout.rv_app_cell, parent))
    }

    override fun onBindContentVHolder(viewHolder: RecyclerView.ViewHolder, data: AppInfo?, position: Int) {
        val contentViewHolder = viewHolder as AppContentHolder
        contentViewHolder.mOnItemClickListener = listener
        contentViewHolder.bindData(data)
    }
}