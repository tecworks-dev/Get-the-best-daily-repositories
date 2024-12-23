package com.lyihub.privacy_radar.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.PixelFormat
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.ColorDrawable
import android.graphics.drawable.Drawable
import android.text.TextUtils
import android.util.Log
import android.widget.ImageView
import androidx.core.graphics.drawable.RoundedBitmapDrawableFactory
import com.bumptech.glide.Glide
import com.bumptech.glide.Priority
import com.bumptech.glide.load.MultiTransformation
import com.bumptech.glide.load.engine.DiskCacheStrategy
import com.bumptech.glide.load.resource.bitmap.BitmapTransformation
import com.bumptech.glide.load.resource.bitmap.CenterCrop
import com.bumptech.glide.load.resource.bitmap.RoundedCorners
import com.bumptech.glide.load.resource.drawable.DrawableTransitionOptions
import com.bumptech.glide.request.RequestOptions
import com.bumptech.glide.request.target.SimpleTarget
import com.bumptech.glide.request.target.Target
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File


class ImageUtils {
    val TAG = "ImageUtils"
    private val animDuration = 500

    private object Holder {
        val instance = ImageUtils()
    }

    private var options: RequestOptions? = getRequestOptions(ColorUtil.getDefaultRandomColor())

    companion object {
        val instance: ImageUtils by lazy { Holder.instance }
    }

    /**
     * 加载头像
     *
     * @param imageView
     * @param url
     */
    fun loadAvatar(context: Context, imageView: ImageView?, url: String?) {
        if (imageView == null)
            return
        if (TextUtils.isEmpty(url)) {
            val colorDrawable = ColorDrawable(ColorUtil.getDefaultRandomColor())
            imageView.setImageDrawable(colorDrawable)
        } else {
            Glide.with(context)
                .load(url)
                .apply(
                    options!!.placeholder(ColorUtil.getDefaultRandomColor())
                        .error(ColorUtil.getDefaultRandomColor())
                )
                .into(object : SimpleTarget<Drawable>() {
                    override fun onResourceReady(resource: Drawable, transition: com.bumptech.glide.request.transition.Transition<in Drawable>?) {
                        if (resource != null) {
                            val bd = resource as BitmapDrawable
                            val circularBitmapDrawable = RoundedBitmapDrawableFactory.create(context.resources, bd.bitmap)
                            circularBitmapDrawable.isCircular = true
                            imageView.setImageDrawable(circularBitmapDrawable)
                        }
                    }
                })
        }
    }

    /**
     * 加载图片
     *
     * @param imageView
     * @param url
     */
    fun loadImage(context: Context, imageView: ImageView?, url: String?): Target<*>? {
        return if (imageView == null) null else Glide.with(context)
            .load(url).apply(
                options!!.placeholder(ColorUtil.getDefaultRandomColor())
                    .error(ColorUtil.getDefaultRandomColor())
            )
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }

    fun loadImage(
        context: Context, imageView: ImageView?, url: String, placeDrawable: Int,
        showPlaceDrawable: Boolean
    ): Target<*>? {
        if (imageView == null)
            return null
        return if (showPlaceDrawable) {
            Glide.with(context)
                .load(url).apply(
                    options!!.placeholder(placeDrawable)
                        .error(placeDrawable)
                )
                .transition(DrawableTransitionOptions().crossFade(animDuration))
                .into(imageView)
        } else Glide.with(context)
            .load(url).apply(options!!.error(placeDrawable))
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }

    fun loadImage(context: Context, imageView: ImageView?, url: String, placeDrawable: Drawable): Target<*>? {
        return if (imageView == null) null else Glide.with(context)
            .load(url).apply(
                options!!.placeholder(ColorUtil.getDefaultRandomColor())
                    .error(placeDrawable)
            )
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }

    fun loadImage(context: Context, imageView: ImageView?, bitmap: Bitmap): Target<*>? {
        return if (imageView == null) null else Glide.with(context)
            .load(bitmap).apply(
                options!!.placeholder(ColorUtil.getDefaultRandomColor())
                    .error(ColorUtil.getDefaultRandomColor())
            )
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }

    fun loadImage(context: Context, imageView: ImageView?, url: String?, errorDrawable: Int): Target<*>? {
        return if (imageView == null) null else Glide.with(context)
            .load(url).apply(
                options!!.placeholder(ColorUtil.getDefaultRandomColor())
                    .error(errorDrawable)
            )
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }

    fun loadImage(context: Context, imageView: ImageView?,width: Int,height: Int,url: String?, errorDrawable: Int): Target<*>? {
        return if (imageView == null) null else Glide.with(context)
            .load(url).apply(
                options!!.placeholder(ColorUtil.getDefaultRandomColor())
                    .error(errorDrawable)
            )
            .override(width, height)
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }

    fun loadImage(context: Context, imageView: ImageView?, file: File?, errorDrawable: Int): Target<*>? {
        return if (imageView == null) null else Glide.with(context)
            .load(file).apply(
                options!!.placeholder(ColorUtil.getDefaultRandomColor())
                    .error(errorDrawable)
            )
            .dontAnimate()
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }
    fun loadImage(context: Context, imageView: ImageView?, bitmap: Bitmap?, errorDrawable: Int): Target<*>? {
        return if (imageView == null) null else Glide.with(context)
            .load(bitmap).apply(
                options!!.placeholder(ColorUtil.getDefaultRandomColor())
                    .error(errorDrawable)
            )
            .dontAnimate()
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }

    fun loadImage(
        context: Context, imageView: ImageView?, url: String?, placeDrawable: Int,
        errorDrawable: Int
    ): Target<*>? {

        return if (imageView == null) null else Glide.with(context)
            .load(url).apply(
                options!!.placeholder(placeDrawable)
                    .error(errorDrawable)
            )
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }

    fun loadImageWithCorner(context: Context, imageView: ImageView, url: String?, corner: Int, transformation: BitmapTransformation = CenterCrop()) {
        Glide.with(context).asBitmap().load(url).apply(
            RequestOptions.bitmapTransform(MultiTransformation(transformation, RoundedCorners(corner)))
        ).into(imageView)
    }

    /**
     * 从文件加入
     *
     * @param imageView
     * @param file
     */
    fun loadImage(context: Context, imageView: ImageView?, file: File) {
        if (imageView == null)
            return
        Glide.with(context)
            .load(file)
            .apply(
                options!!.placeholder(ColorUtil.getDefaultRandomColor())
                    .error(ColorUtil.getDefaultRandomColor())
            )
            .transition(DrawableTransitionOptions().crossFade(animDuration))
            .into(imageView)
    }

    /**
     * 加载图片
     *
     * @param imageView
     * @param drawableRes
     */
    fun loadImage(context: Context, imageView: ImageView?, drawableRes: Int?) {
        if (imageView == null)
            return
        loadImage(context, imageView, drawableRes ?: 0, ColorUtil.getDefaultRandomColor())
    }

    fun loadImage(context: Context, imageView: ImageView?, drawableRes: Int, placeDrawable: Int) {
        if (imageView == null)
            return
        Glide.with(context).load(drawableRes).apply(
            options!!.placeholder(ColorUtil.getDefaultRandomColor())
                .error(placeDrawable)
        ).transition(
            DrawableTransitionOptions()
                .crossFade(animDuration)
        ).into(imageView)
    }

    fun getRequestOptions(resId: Int): RequestOptions? {
        if (options == null) {
            options = RequestOptions()
                .placeholder(resId)
                .error(resId)
                .priority(Priority.HIGH)
                .diskCacheStrategy(DiskCacheStrategy.ALL)
        }
        return options
    }
}