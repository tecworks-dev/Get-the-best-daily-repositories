import App from '../app/App';
import { TestReportInfo } from '../data/TestReportInfo';
import { Constants } from '../util/Constants';
import { lpx2fp } from '../util/DesignShow';
import { ToolBar } from '../widget/ToolBar';

@Entry
@Component
struct TestReportPage {
  @State testReportList: Array<TestReportInfo> = []

  aboutToAppear(): void {
    this.testReportList = App.get().getTestReport()
  }

  build() {
    Column() {
      this.titleBar()
      this.content()

    }
    .width(Constants.FULL_PARENT)
    .height(Constants.FULL_PARENT)
    .backgroundColor($r('app.color.white'))
    .padding({ bottom: $r('app.string.lpx_60') })
  }

  @Builder titleBar() {
    Row() {
      Image($r('app.media.ic_back'))
        .width($r('app.string.lpx_48'))
        .height($r('app.string.lpx_48'))
        .objectFit(ImageFit.Contain)
        .margin({ left: $r('app.string.lpx_32') })
        .onClick(() => {
            App.get().back()
        })

      Column(){
        Text('隐私雷达')
          .fontSize(lpx2fp(36))
          .fontColor($r('app.color.black'))
          .fontWeight(FontWeight.Bold)
          .textAlign(TextAlign.Start)
        Text('by 林亦 LYi')
          .fontSize(lpx2fp(28))
          .fontColor($r('app.color.color_999999'))
      }
      .layoutWeight(1)
      .alignItems(HorizontalAlign.Start)
      .padding({
        top: $r('app.string.lpx_10'),
        bottom: $r('app.string.lpx_10'),
        left: $r('app.string.lpx_24'),
        right: $r('app.string.lpx_24')
      })

    }
    .expandSafeArea([SafeAreaType.KEYBOARD]) // 配置键盘弹出安全区域
    .width(Constants.FULL_PARENT)
    .padding({ top: $r('app.string.lpx_60') })
    .shadow({
      radius: 5,
      color: $r('app.color.color_1400001E'),
      offsetX: 5,
      offsetY: 5
    })
  }

  @Builder content() {
    Text("欢迎来到隐私雷达\n测试已完成")
      .width(Constants.FULL_PARENT)
      .fontColor($r('app.color.black'))
      .fontSize(lpx2fp(56))
      .textAlign(TextAlign.Center)
      .margin({
        top: $r('app.string.lpx_30'),
        left: $r('app.string.lpx_30')
      })

    List({ space: 10 }) {
      ForEach(this.testReportList,(item: TestReportInfo,index: number) => {
        ListItem() {
          this.itemLayout(item,index)
        }
      })
    }
    .width(Constants.FULL_PARENT)
    .layoutWeight(1)
    .alignSelf(ItemAlign.Center)
    .margin({
      top: $r('app.string.lpx_50')
    })
    .padding($r('app.string.lpx_10'))
  }

  @Builder itemLayout(item: TestReportInfo,index: number) {
    Column(){
      Text(`${App.get().getTestResult(item)}`)
        .width(Constants.FULL_PARENT)
        .fontSize(lpx2fp(28))
        .fontColor($r('app.color.black'))
        .maxLines(1)
        .textOverflow({overflow: TextOverflow.Ellipsis})
      Text(`${item.content}`)
        .width(Constants.FULL_PARENT)
        .fontSize(lpx2fp(28))
        .fontColor($r('app.color.black'))
        .maxLines(1)
        .textOverflow({overflow: TextOverflow.Ellipsis})
        .margin({
          top: $r('app.string.lpx_20')
        })
      Line()
        .width(Constants.FULL_PARENT)
        .height($r('app.string.lpx_1'))
        .backgroundColor($r('app.color.color_CCCCCC'))
        .margin({
          top: $r('app.string.lpx_20')
        })
    }
    .width("95%")
    .backgroundColor($r('app.color.white'))
    .justifyContent(FlexAlign.Center)
    .padding({
      top: $r('app.string.lpx_20'),
      left: $r('app.string.lpx_20'),
      right: $r('app.string.lpx_20')
    })
  }
}