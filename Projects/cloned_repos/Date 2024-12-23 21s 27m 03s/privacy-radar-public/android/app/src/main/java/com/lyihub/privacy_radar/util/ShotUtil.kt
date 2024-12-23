package com.hok.lib.common.util

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import com.lyihub.privacy_radar.util.ScreenUtils
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.IOException


object ShotUtil {
    private const val TAG = "ShotUtil"

    /**
     * 屏幕截图
     *
     * @param activity
     * @return
     */
    fun screenShot(activity: Activity?): Bitmap? {
        if (activity == null) {
            Log.e(TAG, "screenShot--->activity is null")
            return null
        }
        val view: View = activity.window.decorView
        //允许当前窗口保存缓存信息
        view.isDrawingCacheEnabled = true
        view.buildDrawingCache()
        val navigationBarHeight = getNavigationBarHeight(view.context)


        //获取屏幕宽和高
        val width: Int = ScreenUtils.getWidth(activity)
        val height: Int = ScreenUtils.getHeight(activity)

        // 全屏不用考虑状态栏，有导航栏需要加上导航栏高度
        var bitmap: Bitmap? = null
        try {
            bitmap = Bitmap.createBitmap(
                view.drawingCache, 0, 0, width,
                height + navigationBarHeight
            )
        } catch (e: Exception) {
            // 这里主要是为了兼容异形屏做的处理，我这里的处理比较仓促，直接靠捕获异常处理
            // 其实vivo oppo等这些异形屏手机官网都有判断方法
            // 正确的做法应该是判断当前手机是否是异形屏，如果是就用下面的代码创建bitmap
            var msg = e.message
            // 部分手机导航栏高度不占窗口高度，不用添加，比如OppoR15这种异形屏
            if (msg!!.contains("<= bitmap.height()")) {
                try {
                    bitmap = Bitmap.createBitmap(
                        view.drawingCache, 0, 0, width,
                        height
                    )
                } catch (e1: Exception) {
                    msg = e1.message
                    // 适配Vivo X21异形屏，状态栏和导航栏都没有填充
                    if (msg!!.contains("<= bitmap.height()")) {
                        try {
                            bitmap = Bitmap.createBitmap(
                                view.drawingCache, 0, 0, width,
                                height - getStatusBarHeight(view.context)
                            )
                        } catch (e2: Exception) {
                            e2.printStackTrace()
                        }
                    } else {
                        e1.printStackTrace()
                    }
                }
            } else {
                e.printStackTrace()
            }
        }

        //销毁缓存信息
        view.destroyDrawingCache()
        view.isDrawingCacheEnabled = false
        return bitmap
    }


    /**
     * view截图
     *
     * @return
     */
    fun viewShot(context: Context, v: View): String? {
        Log.e(TAG, "viewShot()......")
        val filePath: String? = null
        if (null == v) {
            Log.e(TAG, "view is null")
            return filePath
        }
        try {
            // 核心代码start
            val bitmap = Bitmap.createBitmap(v.width, v.height, Bitmap.Config.ARGB_8888)
            val c = Canvas(bitmap)
            v.layout(v.left, v.top, v.right, v.bottom)
            v.draw(c)
            return saveImageToGallery(context, bitmap)
        } catch (e: Exception) {
        }
        return null
    }

    fun takeScreenshot(scannerView: View): Bitmap? {
        // 创建一个和scannerView一样大小的空的Bitmap
        val bitmap = Bitmap.createBitmap(
            scannerView.getWidth(),
            scannerView.getHeight(),
            Bitmap.Config.ARGB_8888
        )
        // 使用Canvas来将scannerView的内容绘制到Bitmap上
        val canvas = Canvas(bitmap)
        scannerView.draw(canvas)
        return bitmap
    }

    fun viewShot(v: View): Bitmap? {
        var bitmap: Bitmap? = null
        Log.e(TAG, "viewShot()......")
        try {
            if (null == v) {
                Log.e(TAG, "view is null")
                return bitmap
            }

            // 核心代码start
            bitmap = Bitmap.createBitmap(v.width, v.height, Bitmap.Config.ARGB_8888)
            val c = Canvas(bitmap)
            v.layout(v.left, v.top, v.right, v.bottom)
            v.isDrawingCacheEnabled = true
            v.draw(c)
            v.isDrawingCacheEnabled = false
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return bitmap
    }

    /**
     * 组装地图截图和其他View截图，需要注意的是目前提供的方法限定为MapView与其他View在同一个ViewGroup下
     * @param    bitmap             地图截图回调返回的结果
     * @param   viewContainer      MapView和其他要截图的View所在的父容器ViewGroup
     * @param   mapView            MapView控件
     * @param   views              其他想要在截图中显示的控件
     */
    /*fun getMapAndViewScreenShot(
        bitmap: Bitmap?,
        viewContainer: ViewGroup,
        mapView: MapView,
        vararg views: View
    ): Bitmap? {
        val width = viewContainer.width
        val height = viewContainer.height
        val screenBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(screenBitmap)
        canvas.drawBitmap(bitmap, mapView.getLeft(), mapView.getTop(), null)
        try {
            for (view in views) {
                view.isDrawingCacheEnabled = true
                canvas.drawBitmap(view.drawingCache, view.left.toFloat(), view.top.toFloat(), null)
                view.isDrawingCacheEnabled = false
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return screenBitmap
    }*/

    /**
     * 保存到系统相册
     *
     * @param context
     * @param bmp
     */
    fun saveImageToGallery(context: Context, bmp: Bitmap?): String? {
        var filePath: String? = null
        Log.e(TAG, "saveImageToGallery()......")
        if (bmp == null) {
            Log.e(TAG, "saveImageToGallery()...bmp == null")
            return filePath
        }
        // 首先保存图片
        val appDir = File(Environment.getExternalStorageDirectory(), "MyCard")
        if (!appDir.exists()) {
            appDir.mkdir()
        }
        val fileName = System.currentTimeMillis().toString() + ".jpg"
        val file = File(appDir, fileName)
        try {
            val fos = FileOutputStream(file)
            bmp.compress(Bitmap.CompressFormat.JPEG, 100, fos)
            fos.flush()
            fos.close()
            filePath = file.absolutePath
        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        } catch (e: IOException) {
            e.printStackTrace()
        }
        // 其次把文件插入到系统图库
         try {
            MediaStore.Images.Media.insertImage(context.getContentResolver(),
                    file.getAbsolutePath(), fileName, null);
        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        }
        Log.e(TAG, "saveImageToGallery()...success")
        // 最后通知图库更新
        context.sendBroadcast(
            Intent(
                Intent.ACTION_MEDIA_SCANNER_SCAN_FILE,
                Uri.parse("file://" + file.absolutePath)
            )
        )
        return filePath
    }

    private fun getStatusBarHeight(context: Context?): Int {
        if (context == null) return 0
        val resources = context.resources
        val resourceId =
            resources.getIdentifier("status_bar_height", "dimen", "android")
        return resources.getDimensionPixelSize(resourceId)
    }

    private fun getNavigationBarHeight(context: Context?): Int {
        if (context == null) return 0
        val resources = context.resources
        val resourceId =
            resources.getIdentifier("navigation_bar_height", "dimen", "android")
        return resources.getDimensionPixelSize(resourceId)
    }
}