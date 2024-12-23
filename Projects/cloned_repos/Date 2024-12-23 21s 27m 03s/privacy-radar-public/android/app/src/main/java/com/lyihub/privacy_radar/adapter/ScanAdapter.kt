package com.lyihub.privacy_radar.adapter

import android.content.Context
import android.graphics.Bitmap
import android.view.ViewGroup
import android.widget.AdapterView
import androidx.recyclerview.widget.RecyclerView
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.holder.AlbumContentHolder
import com.lyihub.privacy_radar.holder.ContactContentHolder
import com.lyihub.privacy_radar.holder.PermissionContentHolder
import com.lyihub.privacy_radar.holder.ScanContentHolder


class ScanAdapter(context: Context, listener: AdapterView.OnItemClickListener?) :
    BaseRecycleAdapter<Bitmap?, RecyclerView.ViewHolder>(context, listener) {

    override fun onCreateHeadVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder? {
        return null
    }

    override fun onBindHeadVHolder(viewHolder: RecyclerView.ViewHolder, data: Bitmap?, position: Int) {
    }

    override fun onCreateContentVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return ScanContentHolder(inflate(R.layout.rv_scan_cell, parent))
    }

    override fun onBindContentVHolder(viewHolder: RecyclerView.ViewHolder, data: Bitmap?, position: Int) {
        val contentViewHolder = viewHolder as ScanContentHolder
        contentViewHolder.mOnItemClickListener = listener
        contentViewHolder.bindData(data)
    }
}