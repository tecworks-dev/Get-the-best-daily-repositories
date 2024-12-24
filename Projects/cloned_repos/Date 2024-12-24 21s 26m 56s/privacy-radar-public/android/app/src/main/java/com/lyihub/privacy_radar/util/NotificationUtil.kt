package com.lyihub.privacy_radar.util

import android.app.Activity
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.lyihub.privacy_radar.AlbumResultActivity
import com.lyihub.privacy_radar.R

object NotificationUtil {
    const val NORMAL_CHANNEL_ID = "NORMAL_CHANNEL_ID"
    const val REQUEST_CODE = 100
    fun sendNotyfy(context: Activity) {
        val notificationManager = ContextCompat.getSystemService(
            context,
            NotificationManager::class.java
        ) as NotificationManager

        // 安卓8.0及以上需要创建渠道
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NORMAL_CHANNEL_ID,
                "CHANNEL_NAME",
                NotificationManager.IMPORTANCE_DEFAULT
            )
            notificationManager.createNotificationChannel(channel)
        }

        // 构建一个PendingIntent
        val intent = Intent(context, AlbumResultActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(context, REQUEST_CODE, intent, PendingIntent.FLAG_IMMUTABLE)

        // 构建Notification配置
        val builder = NotificationCompat.Builder(context, NORMAL_CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("LinYiLab|相册更新了")
            .setContentText("点击查看")
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)

        // 发起通知
        notificationManager.notify(1, builder.build())
    }
}