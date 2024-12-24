import deviceinfo from '@ohos.deviceInfo';
import App from '../app/App';
import { Constants } from '../util/Constants';
import { lpx2fp } from '../util/DesignShow';
import DeviceUtil from '../util/DeviceUtil';
import { ToolBar } from '../widget/ToolBar';

@Entry
@Component
struct DeviceResultPage {

  aboutToAppear(): void {
    App.get().putDeviceResult(1)
  }

  build() {
    Column() {
      ToolBar({
        title: '隐私雷达|手机信息',
        titleFontColor:$r('app.color.black'),
        titleBackgroundColor: $r('app.color.color_primary')
      })
      this.content()
    }
    .width(Constants.FULL_PARENT)
    .height(Constants.FULL_PARENT)
    .backgroundColor($r('app.color.white'))
  }

  @Builder content() {
    Column() {
      Text('获取手机信息权限\n完成')
        .textAlign(TextAlign.Start)
        .width(Constants.FULL_PARENT)
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(56))
        .margin({
          top: $r('app.string.lpx_30'),
          left: $r('app.string.lpx_30')
        })

      Text(`IMEI:${DeviceUtil.getSerial()}`)
        .width(Constants.FULL_PARENT)
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(32))
        .margin({
          top: $r('app.string.lpx_50')
        })

      Text(`S/N:${DeviceUtil.getSerial()}`)
        .width(Constants.FULL_PARENT)
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(32))
      Text(`OsVersion:${deviceinfo.displayVersion}`)
        .width(Constants.FULL_PARENT)
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(32))
      Text(`BuildVersion:${deviceinfo.sdkApiVersion}`)
        .width(Constants.FULL_PARENT)
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(32))
      Text(`Model:${deviceinfo.softwareModel}`)
        .width(Constants.FULL_PARENT)
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(32))
    }
    .width(Constants.FULL_PARENT)
    .layoutWeight(1)
    .alignSelf(ItemAlign.Center)
    .margin({
      top: $r('app.string.lpx_50')
    })
    .padding($r('app.string.lpx_10'))
  }
}