import App from '../app/App'
import { PermissionInfo } from '../data/PermissionInfo'

class DataUtil {

  getMenuList(): Array<PermissionInfo> {
    let menuList = Array<PermissionInfo>()

    let item1 = new PermissionInfo()
    item1.content = $r('app.string.album')
    if (App.get().getAlbumCount() == -1) {
      item1.icon = $r('app.media.ic_unknown')
      item1.status = '尚未获取授权'
    } else {
      item1.icon = $r('app.media.ic_secure')
      item1.status = '无泄漏风险'
    }

    let item2 = new PermissionInfo()
    item2.content = $r('app.string.contact')
    if (App.get().getContactCount() == -1) {
      item2.icon = $r('app.media.ic_unknown')
      item2.status = '尚未获取授权'
    } else {
      item2.icon = $r('app.media.ic_secure')
      item2.status = '无泄漏风险'
    }

    let item3 = new PermissionInfo()
    item3.content = $r('app.string.file')
    if (App.get().getFileCount() == -1) {
      item3.icon = $r('app.media.ic_unknown')
      item3.status = '尚未获取授权'
    } else {
      item3.icon = $r('app.media.ic_secure')
      item3.status = '无泄漏风险'
    }

    let item4 = new PermissionInfo()
    item4.content = $r('app.string.qr_scan')
    if (App.get().getScanResult() == -1) {
      item4.icon = $r('app.media.ic_unknown')
      item4.status = '尚未获取授权'
    } else {
      item4.icon = $r('app.media.ic_secure')
      item4.status = '无泄漏风险'
    }

    let item5 = new PermissionInfo()
    item5.content = $r('app.string.device_info')
    if (App.get().getDeviceResult() == -1) {
      item5.icon = $r('app.media.ic_unknown')
      item5.status = '尚未获取授权'
    } else {
      item5.icon = $r('app.media.ic_secure')
      item5.status = '无泄漏风险'
    }

    let item6 = new PermissionInfo()
    item6.content = $r('app.string.call_log')
    if (App.get().getCallLogCount() == -1) {
      item6.icon = $r('app.media.ic_unknown')
      item6.status = '尚未获取授权'
    } else {
      item6.icon = $r('app.media.ic_secure')
      item6.status = '无泄漏风险'
    }

    let item7 = new PermissionInfo()
    item7.content = $r('app.string.read_sms')
    if (App.get().getSmsCount() == -1) {
      item7.icon = $r('app.media.ic_unknown')
      item7.status = '尚未获取授权'
    } else {
      item7.icon = $r('app.media.ic_secure')
      item7.status = '无泄漏风险'
    }

    let item8 = new PermissionInfo()
    item8.content = $r('app.string.app')
    if (App.get().getAppCount() == -1) {
      item8.icon = $r('app.media.ic_unknown')
      item8.status = '尚未获取授权'
    } else {
      item8.icon = $r('app.media.ic_secure')
      item8.status = '无泄漏风险'
    }

    menuList.push(item1)
    menuList.push(item2)
    menuList.push(item3)
    menuList.push(item4)
    menuList.push(item5)
    menuList.push(item6)
    menuList.push(item7)
    menuList.push(item8)

    return menuList
  }

  getIcon(position: number): Resource {
    let iconRes = $r('app.media.ic_unknown')
    if (position == 0) {//相册
      if (App.get().getAlbumCount() == -1) {
        iconRes = $r('app.media.ic_unknown')
      } else {
        iconRes = $r('app.media.ic_secure')
      }
    } else if (position == 1) {//联系人
      if (App.get().getContactCount() == -1) {
        iconRes = $r('app.media.ic_unknown')
      } else {
        iconRes = $r('app.media.ic_secure')
      }
    } else if (position == 2) {//文件
      if (App.get().getFileCount() == -1) {
        iconRes = $r('app.media.ic_unknown')
      } else {
        iconRes = $r('app.media.ic_secure')
      }
    } else if (position == 3) {//相机
      if (App.get().getScanResult() == -1) {
        iconRes = $r('app.media.ic_unknown')
      } else {
        iconRes = $r('app.media.ic_secure')
      }
    } else if (position == 4) {//手机信息
      if (App.get().getDeviceResult() == -1) {
        iconRes = $r('app.media.ic_unknown')
      } else {
        iconRes = $r('app.media.ic_secure')
      }
    } else if (position == 5) {//通话记录
      if (App.get().getCallLogCount() == -1) {
        iconRes = $r('app.media.ic_unknown')
      } else {
        iconRes = $r('app.media.ic_secure')
      }
    } else if (position == 6) {//短信权限
      if (App.get().getSmsCount() == -1) {
        iconRes = $r('app.media.ic_unknown')
      } else {
        iconRes = $r('app.media.ic_secure')
      }
    } else if (position == 7) {//应用列表
      if (App.get().getAppCount() == -1) {
        iconRes = $r('app.media.ic_unknown')
      } else {
        iconRes = $r('app.media.ic_secure')
      }
    }
    return iconRes
  }

  getStatus(position: number): string {
    let status = '尚未获取授权'
    if (position == 0) {//相册
      if (App.get().getAlbumCount() == -1) {
        status = '尚未获取授权'
      } else {
        status = '无泄漏风险'
      }
    } else if (position == 1) {//联系人
      if (App.get().getContactCount() == -1) {
        status = '尚未获取授权'
      } else {
        status = '无泄漏风险'
      }
    } else if (position == 2) {//文件
      if (App.get().getFileCount() == -1) {
        status = '尚未获取授权'
      } else {
        status = '无泄漏风险'
      }
    } else if (position == 3) {//相机
      if (App.get().getScanResult() == -1) {
        status = '尚未获取授权'
      } else {
        status = '无泄漏风险'
      }
    } else if (position == 4) {//手机信息
      if (App.get().getDeviceResult() == -1) {
        status = '尚未获取授权'
      } else {
        status = '无泄漏风险'
      }
    } else if (position == 5) {//通话记录
      if (App.get().getCallLogCount() == -1) {
        status = '尚未获取授权'
      } else {
        status = '无泄漏风险'
      }
    } else if (position == 6) {//短信权限
      if (App.get().getSmsCount() == -1) {
        status = '尚未获取授权'
      } else {
        status = '无泄漏风险'
      }
    } else if (position == 7) {//应用列表
      if (App.get().getAppCount() == -1) {
        status = '尚未获取授权'
      } else {
        status = '无泄漏风险'
      }
    }
    return status
  }


}

export default new DataUtil()