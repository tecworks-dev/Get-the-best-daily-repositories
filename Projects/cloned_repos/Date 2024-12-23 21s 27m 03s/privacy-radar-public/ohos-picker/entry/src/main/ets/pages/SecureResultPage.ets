import { router } from '@kit.ArkUI'
import App from '../app/App'
import { Constants } from '../util/Constants'
import { lpx2fp } from '../util/DesignShow'
import { ToolBar } from '../widget/ToolBar'

@Entry
@Component
struct SecureResultPage {
  @State tip: string = ''
  @State title: string = ''
  @State type: number = 0

  aboutToAppear(): void {
    this.tip = (router.getParams() as object)?.[Constants.DATA]
    this.title = (router.getParams() as object)?.[Constants.TITLE]
    this.type = (router.getParams() as object)?.[Constants.TYPE]
    if (this.type == 1) {
      App.get().putCallLogCount(1)
    } else if (this.type == 2) {
      App.get().putSmsCount(1)
    } else if (this.type == 3) {
      App.get().putAppCount(1)
    }

  }

  build() {
    Column() {
      ToolBar({
        title: this.title,
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
    Text(this.tip)
      .fontColor($r('app.color.black'))
      .fontSize(lpx2fp(56))
      .textAlign(TextAlign.Start)
      .width(Constants.FULL_PARENT)
      .margin({
        top: $r('app.string.lpx_50'),
        left: $r('app.string.lpx_30')
      })

    Text(){
      ImageSpan($r('app.media.ic_secure'))
        .width($r('app.string.lpx_80'))
        .height($r('app.string.lpx_80'))
        .verticalAlign(ImageSpanAlignment.CENTER)
      Span('无泄漏风险')
        .fontColor($r('app.color.color_43CD80'))
        .fontSize(lpx2fp(36))
        .fontWeight(FontWeight.Bold)
    }
    .width(Constants.FULL_PARENT)
    .margin({
      top: $r('app.string.lpx_30'),
      left: $r('app.string.lpx_30')
    })
  }

  @Builder itemLayout(item: string,index: number) {
    Column(){
      Image(item)
        .width(Constants.FULL_PARENT)
        .height(Constants.FULL_PARENT)
    }
    .width("95%")
    .height($r('app.string.lpx_250'))
    .backgroundColor($r('app.color.color_33EB4F3A'))
    .justifyContent(FlexAlign.Center)
  }

}