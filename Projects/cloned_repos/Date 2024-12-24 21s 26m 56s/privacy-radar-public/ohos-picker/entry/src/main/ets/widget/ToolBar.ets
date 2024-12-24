import App from '../app/App'
import { Constants } from '../util/Constants'
import { lpx2fp } from '../util/DesignShow'

@Preview
@Component
export struct ToolBar {
  @State title?: string = ''
  @State titleFontColor?: Resource = $r('app.color.color_333333')
  @State titleBackgroundColor?: Resource = $r('app.color.white')
  @State titleBackIcon?: Resource | null = $r('app.media.ic_back')
  @State titleActionIcon?: Resource = $r('app.color.transparent')
  @State titleBackEnable: boolean = true
  @State shadowColor?: Resource = $r('app.color.color_1400001E')
  titleActionClick: () => void = () => {
  }

  build() {
    Row() {
      if (this.titleBackEnable) {
        Image(this.titleBackIcon)
          .width($r('app.string.lpx_48'))
          .height($r('app.string.lpx_48'))
          .objectFit(ImageFit.Contain)
          .margin({ left: $r('app.string.lpx_32') })
          .onClick(() => {
            if (this.titleBackEnable) {
              App.get().back()
            }
          })
      }

      Text(this.title)
        .layoutWeight(1)
        .height($r('app.string.lpx_98'))
        .fontSize(lpx2fp(36))
        .fontColor(this.titleFontColor)
        .fontWeight(FontWeight.Bold)
        .textAlign(TextAlign.Start)
        .padding({
          left: $r('app.string.lpx_24'),
          right: $r('app.string.lpx_24')
        })

      Image(this.titleActionIcon)
        .width($r('app.string.lpx_48'))
        .height($r('app.string.lpx_48'))
        .objectFit(ImageFit.Contain)
        .margin({ right: $r('app.string.lpx_32') })
        .onClick(() => {
          this.titleActionClick()
        })
    }
    .expandSafeArea([SafeAreaType.KEYBOARD]) // 配置键盘弹出安全区域
    .width(Constants.FULL_PARENT)
    .backgroundColor(this.titleBackgroundColor)
    .padding({ top: $r('app.string.lpx_60') })
    .shadow({
      radius: 5,
      color: this.shadowColor,
      offsetX: 5,
      offsetY: 5
    })
    .zIndex(1)
  }
}