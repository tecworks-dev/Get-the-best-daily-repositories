package com.lyihub.privacy_radar.util

import android.app.Activity
import android.os.Environment
import android.util.Log
import androidx.fragment.app.Fragment
import com.luck.picture.lib.basic.PictureSelector
import com.luck.picture.lib.config.PictureConfig
import com.luck.picture.lib.config.PictureMimeType
import com.luck.picture.lib.config.SelectMimeType
import com.luck.picture.lib.config.SelectModeConfig
import com.luck.picture.lib.entity.LocalMedia
import com.luck.picture.lib.style.PictureSelectorStyle
import java.io.File


object PictureSelectorUtil {
    private const val TAG = "PictureSelectorUtil"
    fun selectMedia(activity: Activity?, isSelectVideo: Boolean, crop: Boolean,isAvatar: Boolean, maxSelectCount: Int) {
        var maxSelectCount = maxSelectCount
        val pictureSelector: PictureSelector? = getPictureSelector(activity, null)
        val singleSelect = maxSelectCount == 1
        if (singleSelect) {
            maxSelectCount = 2
        }
        configPictureSelector(pictureSelector, true,isSelectVideo, singleSelect, crop,isAvatar, maxSelectCount)
    }
    fun selectMedia(activity: Activity?,isCamera: Boolean, isSelectVideo: Boolean, crop: Boolean,isAvatar: Boolean, maxSelectCount: Int) {
        var maxSelectCount = maxSelectCount
        val pictureSelector: PictureSelector? = getPictureSelector(activity, null)
        val singleSelect = maxSelectCount == 1
        if (singleSelect) {
            maxSelectCount = 2
        }
        configPictureSelector(pictureSelector, isCamera,isSelectVideo, singleSelect, crop,isAvatar, maxSelectCount)
    }

    fun selectMedia(fragment: Fragment?, isSelectVideo: Boolean, crop: Boolean,isAvatar: Boolean, maxSelectCount: Int) {
        var maxSelectCount = maxSelectCount
        val pictureSelector: PictureSelector? = getPictureSelector(null, fragment)
        val singleSelect = maxSelectCount == 1
        if (singleSelect) {
            maxSelectCount = 2
        }
        configPictureSelector(pictureSelector, true,isSelectVideo, singleSelect, crop,isAvatar, maxSelectCount)
    }
    fun selectMedia(fragment: Fragment?, isCamera: Boolean,isSelectVideo: Boolean, crop: Boolean,isAvatar: Boolean, maxSelectCount: Int) {
        var maxSelectCount = maxSelectCount
        val pictureSelector: PictureSelector? = getPictureSelector(null, fragment)
        val singleSelect = maxSelectCount == 1
        if (singleSelect) {
            maxSelectCount = 2
        }
        configPictureSelector(pictureSelector,isCamera, isSelectVideo, singleSelect, crop,isAvatar, maxSelectCount)
    }

    fun openCamera(activity: Activity?, isSelectVideo: Boolean,crop: Boolean) {
        val pictureSelector: PictureSelector? = getPictureSelector(activity, null)
        configCamera(pictureSelector,isSelectVideo,true,crop,1)
    }
    fun openCamera(fragment: Fragment?, isSelectVideo: Boolean,crop: Boolean) {
        val pictureSelector: PictureSelector? = getPictureSelector(null, fragment)
        configCamera(pictureSelector,isSelectVideo,true,crop,1)
    }

    private fun getPictureSelector(activity: Activity?, fragment: Fragment?): PictureSelector? {
        // 进入相册 以下是例子：不需要的api可以不写
        var pictureSelector: PictureSelector? = null
        if (activity != null) {
            pictureSelector = PictureSelector.create(activity)
        }
        if (fragment != null) {
            pictureSelector = PictureSelector.create(fragment)
        }
        return pictureSelector
    }

    private fun configPictureSelector(pictureSelector: PictureSelector?,
                                      isCamera: Boolean,
                                      isSelectVideo: Boolean, singleSelect: Boolean,
                                      crop: Boolean, isAvatar: Boolean,maxSelectCount: Int) {
        if (pictureSelector == null) return
        pictureSelector.openGallery(if (isSelectVideo) SelectMimeType.ofVideo() else SelectMimeType.ofImage()) // 全部.PictureMimeType.ofAll()、图片.ofImage()、视频.ofVideo()、音频.ofAudio()
            .setSelectorUIStyle(PictureSelectorStyle())
            .setImageEngine(GlideEngine.instance) // 请参考Demo GlideEngine.java
            .setMaxSelectNum(maxSelectCount) // 最大图片选择数量
            .setMinSelectNum(1) // 最小选择数量
            .setMaxVideoSelectNum(maxSelectCount)
            .setImageSpanCount(4) // 每行显示个数
            .setSelectionMode(if (singleSelect) SelectModeConfig.SINGLE else SelectModeConfig.MULTIPLE) // 多选 or 单选
            .isPreviewImage(true) // 是否可预览图片
            .isDisplayCamera(isCamera) // 是否显示拍照按钮
            .isSelectZoomAnim(true) // 图片列表点击 缩放效果 默认true
//                .setCropEngine(ImageFileCropEngine()) // 是否裁剪
            .forResult(PictureConfig.CHOOSE_REQUEST) //结果回调onActivityResult code
    }
    private fun configCamera(pictureSelector: PictureSelector?,
                                      isSelectVideo: Boolean, singleSelect: Boolean,
                                      crop: Boolean, maxSelectCount: Int) {
        if (pictureSelector == null) return
        pictureSelector.openCamera(if (isSelectVideo) SelectMimeType.ofVideo() else SelectMimeType.ofImage()) // 全部.PictureMimeType.ofAll()、图片.ofImage()、视频.ofVideo()、音频.ofAudio()
            .forResultActivity(PictureConfig.REQUEST_CAMERA) //结果回调onActivityResult code
    }

    fun getPath(): String? {
        val path = Environment.getExternalStorageState().toString() + "/Luban/image/"
        val file = File(path)
        return if (file.mkdirs()) {
            path
        } else path
    }

    fun isVideo(media: LocalMedia?): Boolean {
        var isSelectVideo = false
        if (media == null) return isSelectVideo
        isSelectVideo = PictureMimeType.isHasVideo(media.getMimeType())
        Log.e(TAG, "isVideo()-isSelectVideo = $isSelectVideo")
        return isSelectVideo
    }
}