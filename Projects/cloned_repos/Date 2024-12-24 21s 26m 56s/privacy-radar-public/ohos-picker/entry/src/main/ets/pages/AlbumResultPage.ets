import { Constants } from '../util/Constants';
import { router } from '@kit.ArkUI';
import { ToolBar } from '../widget/ToolBar';
import { lpx2fp } from '../util/DesignShow';
import App from '../app/App';

@Entry
@Component
struct AlbumResultPage {
  @State albumList: Array<string> = []

  aboutToAppear(): void {
    this.albumList = (router.getParams() as object)?.[Constants.DATA]
    App.get().addAlbumList(this.albumList)
  }

  build() {
    Column() {
      ToolBar({
        title: '隐私雷达|相册权限',
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
    Text(){
      Span(`获取相册\n${this.albumList.length}`)
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(56))
      Span('张')
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(28))
    }
    .textAlign(TextAlign.Start)
    .width(Constants.FULL_PARENT)
    .margin({
      top: $r('app.string.lpx_30'),
      left: $r('app.string.lpx_30')
    })

    List({ space: 10 }) {
      ForEach(this.albumList,(item: string,index: number) => {
        ListItem() {
          this.itemLayout(item,index)
        }
      })
    }
    .width(Constants.FULL_PARENT)
    .layoutWeight(1)
    .alignSelf(ItemAlign.Center)
    .lanes(3)
    .margin({
      top: $r('app.string.lpx_50')
    })
    .padding($r('app.string.lpx_10'))
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