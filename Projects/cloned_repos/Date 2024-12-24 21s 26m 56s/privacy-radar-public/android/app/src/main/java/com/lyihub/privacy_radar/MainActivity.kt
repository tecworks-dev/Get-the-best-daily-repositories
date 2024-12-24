package com.lyihub.privacy_radar

import android.Manifest
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.View.OnClickListener
import android.widget.AdapterView
import android.widget.AdapterView.OnItemClickListener
import android.widget.Toast
import com.cherry.permissions.lib.EasyPermissions
import com.cherry.permissions.lib.EasyPermissions.hasStoragePermission
import com.cherry.permissions.lib.annotations.AfterPermissionGranted
import com.cherry.permissions.lib.dialogs.DEFAULT_SETTINGS_REQ_CODE
import com.cherry.permissions.lib.dialogs.SettingsDialog
import com.google.android.material.snackbar.Snackbar
import com.lyihub.privacy_radar.adapter.PermissionAdapter
import com.lyihub.privacy_radar.app.App
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.data.PermissionInfo
import com.lyihub.privacy_radar.util.PermissionRequestCode.REQUEST_CODE_CAMERA_PERMISSION
import com.lyihub.privacy_radar.util.PermissionRequestCode.REQUEST_CODE_CONTACTS_PERMISSION
import com.lyihub.privacy_radar.util.PermissionRequestCode.REQUEST_CODE_LOCATION_AND_CONTACTS_PERMISSION
import com.lyihub.privacy_radar.util.PermissionRequestCode.REQUEST_CODE_READ_CALL_LOG_PERMISSION
import com.lyihub.privacy_radar.util.PermissionRequestCode.REQUEST_CODE_READ_PHONE_STATE_PERMISSION
import com.lyihub.privacy_radar.util.PermissionRequestCode.REQUEST_CODE_READ_SMS_PERMISSION
import com.lyihub.privacy_radar.util.PermissionRequestCode.REQUEST_CODE_STORAGE_PERMISSION
import com.lyihub.privacy_radar.util.ResUtils
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import com.lyihub.privacy_radar.util.TestResultUtil
import kotlinx.android.synthetic.main.activity_main.*


class MainActivity : BaseActivity(),OnClickListener, EasyPermissions.PermissionCallbacks,
    EasyPermissions.RationaleCallbacks,OnItemClickListener {

    private val REQUEST_QR_CODE = 100
    var mPermissionAdapter: PermissionAdapter? = null

    override fun getLayoutResource() = R.layout.activity_main

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData()
    }

    override fun onResume() {
        super.onResume()
        if (TestResultUtil.isTestCompleted()) {
            mTvTip.text = "欢迎来到隐私雷达\n测试已完成"
            mTvTestResult.visibility = View.VISIBLE
        } else {
            mTvTip.text = "欢迎来到隐私雷达\n请完成测试"
            mTvTestResult.visibility = View.GONE
        }
        mPermissionAdapter?.notifyDataSetChanged()
    }

    fun initView() {
        mPermissionAdapter = PermissionAdapter(this,this)
        mRvPermission.adapter = mPermissionAdapter

        mTvTestResult.setOnClickListener(this)
    }

    fun initData() {
        App.get().hasAlbumPermission = hasStoragePermission(this)
        App.get().hasContactsPermissions = hasContactsPermissions()
        App.get().hasCameraPermission = hasCameraPermission()
        App.get().hasCallLogPermission = hasCallLogPermission()
        App.get().hasSmsPermission = hasSmsPermission()

        var playSpeeds = ResUtils.getStringArrayRes(R.array.permission_titles)
        playSpeeds?.forEach {
            val item = PermissionInfo()
            item.title = it
            mPermissionAdapter?.add(item)
        }
        mPermissionAdapter?.notifyDataSetChanged()
    }

    override fun onClick(v: View) {
        when (v?.id) {
            R.id.mTvTestResult -> {
                TestReportActivity.intentStart(this)
            }
        }
    }

    @AfterPermissionGranted(REQUEST_CODE_CAMERA_PERMISSION)
    private fun requestPermissionCamera() {
        if (hasCameraPermission()) {
            App.get().hasCameraPermission = true
            // Have permission, do things!
//            showMessage(mRvPermission,"AfterPermissionGranted you have Camera permission,you can take photo")
            openCameraAction()
        } else {
            // Ask for one permission
            EasyPermissions.requestPermissions(
                this,
                getString(R.string.permission_camera_rationale_message),
                REQUEST_CODE_CAMERA_PERMISSION,
                Manifest.permission.CAMERA
            )
        }
    }

    @AfterPermissionGranted(REQUEST_CODE_READ_CALL_LOG_PERMISSION)
    private fun requestPermissionReadCallLog() {
        if (hasCallLogPermission()) {
            App.get().hasCallLogPermission = true
            // Have permission, do things!
//            showMessage(mRvPermission,"AfterPermissionGranted you have call log permission,you can get call log")
            CallLogResultActivity.intentStart(this)
        } else {
            // Ask for one permission
            EasyPermissions.requestPermissions(
                this,
                getString(R.string.permission_read_call_log_rationale_message),
                REQUEST_CODE_READ_CALL_LOG_PERMISSION,
                Manifest.permission.READ_CALL_LOG
            )
        }
    }

    @AfterPermissionGranted(REQUEST_CODE_READ_SMS_PERMISSION)
    private fun requestPermissionReadSms() {
        if (hasReadSmsPermission()) {
            App.get().hasSmsPermission = true
            // Have permission, do things!
//            showMessage(mRvPermission,"AfterPermissionGranted you have call log permission,you can get call log")
            SmsResultActivity.intentStart(this)
        } else {
            // Ask for one permission
            EasyPermissions.requestPermissions(
                this,
                getString(R.string.permission_read_sms_rationale_message),
                REQUEST_CODE_READ_SMS_PERMISSION,
                Manifest.permission.READ_SMS
            )
        }
    }

    @AfterPermissionGranted(REQUEST_CODE_CONTACTS_PERMISSION)
    private fun requestContactsPermission() {
        if (hasContactsPermissions()) {
            App.get().hasContactsPermissions = true
            // Have permissions, do things!
//            showMessage(mRvPermission,"AfterPermissionGranted you have Contacts permissions,you can get Contacts")
            ContactResultActivity.intentStart(this)
        } else {
            // Ask for both permissions
            EasyPermissions.requestPermissions(
                this,
                getString(R.string.permission_contacts_rationale_message),
                REQUEST_CODE_CONTACTS_PERMISSION,
                Manifest.permission.READ_CONTACTS
            )
        }
    }

    @AfterPermissionGranted(REQUEST_CODE_STORAGE_PERMISSION)
    private fun requestStoragePermission() {
        if (hasStoragePermission(this)) {
            App.get().hasAlbumPermission = true
            // Have permission, do things!
//            showMessage(mRvPermission,"AfterPermissionGranted you have Storage permission,you can storage things")
            AlbumResultActivity.intentStart(this)
        } else {
            // Ask for one permission
            EasyPermissions.requestStoragePermission(
                this,
                getString(R.string.permission_storage_rationale_message),
                REQUEST_CODE_STORAGE_PERMISSION
            )
        }
    }

    private fun hasCameraPermission(): Boolean {
        return EasyPermissions.hasPermissions(this, Manifest.permission.CAMERA)
    }

    private fun hasCallLogPermission(): Boolean {
        return EasyPermissions.hasPermissions(this, Manifest.permission.READ_CALL_LOG)
    }
    private fun hasReadSmsPermission(): Boolean {
        return EasyPermissions.hasPermissions(this, Manifest.permission.READ_SMS)
    }

    private fun hasLocationAndContactsPermissions(): Boolean {
        return EasyPermissions.hasPermissions(this,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.READ_CONTACTS
        )
    }
    private fun hasContactsPermissions(): Boolean {
        return EasyPermissions.hasPermissions(this,
            Manifest.permission.READ_CONTACTS
        )
    }

    private fun hasSmsPermission(): Boolean {
        return EasyPermissions.hasPermissions(this, Manifest.permission.READ_SMS)
    }

    fun showMessage(view: View,message: String) {
        Snackbar.make(view, message, Snackbar.LENGTH_LONG)
            .setAction("Action", null).show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode != RESULT_OK) return
        when (requestCode) {
            DEFAULT_SETTINGS_REQ_CODE -> {
                val yes = getString(R.string.yes)
                val no = getString(R.string.no)

                // Do something after user returned from app settings screen, like showing a Toast.
                Toast.makeText(
                    this,
                    getString(
                        R.string.returned_from_app_settings_to_activity,
                        if (hasCameraPermission()) yes else no,
                        if (hasLocationAndContactsPermissions()) yes else no,
                        if (hasSmsPermission()) yes else no,
                        if (hasStoragePermission(this)) yes else no
                    ),
                    Toast.LENGTH_LONG
                ).show()
            }
            REQUEST_QR_CODE -> {

            }
        }
        if (requestCode == DEFAULT_SETTINGS_REQ_CODE) {
            val bundle = data?.extras
            if (bundle != null) {
                val result = bundle.getString("result")
                Toast.makeText(applicationContext, result, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        // EasyPermissions handles the request result.
        EasyPermissions.onRequestPermissionsResult(requestCode, permissions, grantResults, this)
    }

    // ============================================================================================
    //  Implementation Permission Callbacks
    // ============================================================================================

    override fun onPermissionsGranted(requestCode: Int, perms: List<String>) {
        Log.d(TAG, getString(R.string.log_permissions_granted, requestCode, perms.size))
        //会回调 AfterPermissionGranted注解对应方法
    }

    override fun onPermissionsDenied(requestCode: Int, perms: List<String>) {
        Log.d(TAG, getString(R.string.log_permissions_denied, requestCode, perms.size))

        // (Optional) Check whether the user denied any permissions and checked "NEVER ASK AGAIN."
        // This will display a dialog directing them to enable the permission in app settings.
        if (EasyPermissions.somePermissionPermanentlyDenied(this, perms)) {

            val settingsDialogBuilder = SettingsDialog.Builder(this)

            when(requestCode) {
                REQUEST_CODE_STORAGE_PERMISSION -> {
                    settingsDialogBuilder.title = getString(
                        com.cherry.permissions.lib.R.string.title_settings_dialog,
                        "Storage Permission")
                    settingsDialogBuilder.rationale = getString(
                        com.cherry.permissions.lib.R.string.rationale_ask_again,
                        "Storage Permission")
                }
                REQUEST_CODE_LOCATION_AND_CONTACTS_PERMISSION -> {
                    settingsDialogBuilder.title = getString(
                        com.cherry.permissions.lib.R.string.title_settings_dialog,
                        "Location and Contacts Permissions")
                    settingsDialogBuilder.rationale = getString(
                        com.cherry.permissions.lib.R.string.rationale_ask_again,
                        "Location and Contacts Permissions")
                }
                REQUEST_CODE_CONTACTS_PERMISSION -> {
                    settingsDialogBuilder.title = getString(
                        com.cherry.permissions.lib.R.string.title_settings_dialog,
                        "Contacts Permissions")
                    settingsDialogBuilder.rationale = getString(
                        com.cherry.permissions.lib.R.string.rationale_ask_again,
                        "Contacts Permissions")
                }
                REQUEST_CODE_CAMERA_PERMISSION -> {
                    /*settingsDialogBuilder.title = getString(
                        com.cherry.permissions.lib.R.string.title_settings_dialog,
                        "Camera Permission")
                    settingsDialogBuilder.rationale = getString(
                        com.cherry.permissions.lib.R.string.rationale_ask_again,
                        "Camera Permission")*/

                    openCameraAction()
                }
            }

            settingsDialogBuilder.build().show()
        }

    }

    // ============================================================================================
    //  Implementation Rationale Callbacks
    // ============================================================================================

    override fun onRationaleAccepted(requestCode: Int) {
        Log.d(TAG, getString(R.string.log_permission_rationale_accepted, requestCode))
    }

    override fun onRationaleDenied(requestCode: Int) {
        Log.d(TAG, getString(R.string.log_permission_rationale_denied, requestCode))
    }

    fun openCameraAction() {
        startActivity(Intent(this,CameraActivity::class.java))
//        PictureSelectorUtil.openCamera(this,false,false)
//        val intent = Intent(this@MainActivity, CaptureActivity::class.java)
//        startActivityForResult(intent, REQUEST_QR_CODE)
    }


    override fun onItemClick(p0: AdapterView<*>?, v: View?, position: Int, id: Long) {
        when(v?.id) {
            R.id.mClPermissionCell -> {
                itemClickAction(position)
            }
        }
    }

    fun itemClickAction(position: Int) {
        when (position) {
            0 -> {//相册
                requestStoragePermission()
            }
            1 -> {//联系人
                requestContactsPermission()
            }
            2 -> {//文件权限
//                requestStoragePermission()
                FileResultActivity.intentStart(this)
            }
            3 -> {//相机权限
                requestPermissionCamera()
            }
            4 -> {//手机信息
//                hasReadPhoneStatePermission()
                DeviceResultActivity.intentStart(this)
            }
            5 -> {//通话记录
                requestPermissionReadCallLog()
            }
            6 -> {//短信
                requestPermissionReadSms()
            }
            7 -> {//应用列表
                AppResultActivity.intentStart(this)
            }
        }
    }
}