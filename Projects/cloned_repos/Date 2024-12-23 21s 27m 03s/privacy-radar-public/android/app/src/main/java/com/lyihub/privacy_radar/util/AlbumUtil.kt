package com.lyihub.privacy_radar.util

import android.content.Context
import android.provider.MediaStore
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch



object AlbumUtil {
    const val TAG = "AlbumUtil"
    fun getAllAlbumImages(context: Context,callback: (List<String>) -> Unit) {
        try {
            val listOfAllImages = ArrayList<String>()
            CoroutineScope(Dispatchers.IO).launch {
                val uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                val projection = arrayOf(
                    MediaStore.Images.Media._ID,
                    MediaStore.Images.Media.DISPLAY_NAME,
                    MediaStore.Images.Media.DATA
                )
                val sortOrder = MediaStore.Images.Media._ID + " DESC"
                val cursor = context.contentResolver.query(uri, projection, null, null, sortOrder)

                var column_index_data = cursor?.getColumnIndexOrThrow(MediaStore.MediaColumns.DATA) ?: 0
                while (cursor?.moveToNext() == true) {
                    var absolutePathOfImage = cursor.getString(column_index_data)
                    Log.e(TAG,"getAlbumCount-absolutePathOfImage = $absolutePathOfImage")
                    listOfAllImages.add(absolutePathOfImage)
                }
                cursor?.close()

                CoroutineScope(Dispatchers.Main).launch {
                    callback.invoke(listOfAllImages)
                    Log.e(TAG,"getAlbumCount-count = ${listOfAllImages.size}")
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}