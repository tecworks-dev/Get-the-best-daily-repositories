import { router } from '@kit.ArkUI';
import App from '../app/App';
import { Constants } from '../util/Constants';
import { lpx2fp } from '../util/DesignShow';
import FileUtil from '../util/FileUtil';
import { ToolBar } from '../widget/ToolBar';

@Entry
@Component
struct FileResultPage {
  @State fileList: Array<string> = []

  aboutToAppear(): void {
    this.fileList = (router.getParams() as object)?.[Constants.DATA]
    App.get().addFileList(this.fileList)
  }

  build() {
    Column() {
      ToolBar({
        title: '隐私雷达 | 文件权限',
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
      Span(`获取文件\n${this.fileList.length}`)
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(56))
      Span('个')
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
      ForEach(this.fileList,(item: string,index: number) => {
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

  @Builder itemLayout(item: string,index: number) {
    Column(){
      Text(FileUtil.getFileNameByPath(item))
        .width(Constants.FULL_PARENT)
        .fontSize(lpx2fp(28))
        .fontColor($r('app.color.black'))
        .maxLines(1)
        .textOverflow({overflow: TextOverflow.Ellipsis})
      Text(item)
        .width(Constants.FULL_PARENT)
        .fontSize(lpx2fp(28))
        .fontColor($r('app.color.black'))
        .maxLines(1)
        .textOverflow({overflow: TextOverflow.Ellipsis})
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