package com.lyihub.privacy_radar.holder

import android.view.View
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.app.App
import com.lyihub.privacy_radar.data.PermissionInfo
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import kotlinx.android.synthetic.main.rv_permission_cell.view.*


class PermissionContentHolder(itemView: View) : ContentViewHolder(itemView) {

    fun bindData(data: PermissionInfo?) {
        itemView.mTvPermission.text = data?.title ?: ""
        when (bindingAdapterPosition) {
            0 -> {
                val hasAlbumPermission = App.get().hasAlbumPermission
                if (hasAlbumPermission) {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_risk)
                    itemView.mTvStatus.text = "有泄漏风险"
                } else {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_unknown)
                    itemView.mTvStatus.text = "尚未获取授权"
                }
            }
            1 -> {
                val hasContactsPermissions = App.get().hasContactsPermissions
                if (hasContactsPermissions) {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_risk)
                    itemView.mTvStatus.text = "有泄漏风险"
                } else {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_unknown)
                    itemView.mTvStatus.text = "尚未获取授权"
                }
            }
            2 -> {
                itemView.mIvStatus.setImageResource(R.mipmap.ic_risk)
                itemView.mTvStatus.text = "有泄漏风险"
            }
            3 -> {
                val hasCameraPermission = App.get().hasCameraPermission
                if (hasCameraPermission) {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_risk)
                    itemView.mTvStatus.text = "有泄漏风险"
                } else {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_unknown)
                    itemView.mTvStatus.text = "尚未获取授权"
                }
            }
            4 -> {
                itemView.mIvStatus.setImageResource(R.mipmap.ic_risk)
                itemView.mTvStatus.text = "有泄漏风险"
            }
            5 -> {
                val hasCallLogPermission = App.get().hasCallLogPermission
                if (hasCallLogPermission) {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_risk)
                    itemView.mTvStatus.text = "有泄漏风险"
                } else {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_unknown)
                    itemView.mTvStatus.text = "尚未获取授权"
                }
            }
            6 -> {
                val hasSmsPermission = App.get().hasSmsPermission
                if (hasSmsPermission) {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_risk)
                    itemView.mTvStatus.text = "有泄漏风险"
                } else {
                    itemView.mIvStatus.setImageResource(R.mipmap.ic_unknown)
                    itemView.mTvStatus.text = "尚未获取授权"
                }
            }
            7 -> {
                itemView.mIvStatus.setImageResource(R.mipmap.ic_risk)
                itemView.mTvStatus.text = "有泄漏风险"
            }
        }
    }

    override fun onLongClick(v: View): Boolean {
        return false
    }
}