import { Constants } from '../util/Constants';
import { lpx2fp } from '../util/DesignShow';

@CustomDialog
export default struct CommonTipDialog {
  @State title: string = ''
  controller: CustomDialogController;
  confirm: () => void = () => {
  };

  build() {
    Column() {
      Text(this.title)
        .width(Constants.FULL_PARENT)
        .height($r('app.string.lpx_128'))
        .fontColor($r('app.color.color_333333'))
        .fontSize(lpx2fp(32))
        .textAlign(TextAlign.Center)

      Divider().height($r('app.string.lpx_1')).strokeWidth(`${$r('app.string.lpx_1')}`).color($r('app.color.color_EFEFEF'))

      Text('确定')
        .width(Constants.FULL_PARENT)
        .height($r('app.string.lpx_80'))
        .fontColor($r('app.color.color_ED1B24'))
        .fontSize(lpx2fp(32))
        .textAlign(TextAlign.Center)
        .onClick(() => {
          this.controller.close()
          this.confirm()
        })
    }
    .width('80%')
    .borderRadius($r('app.string.lpx_8'))
    .backgroundColor(Color.White)
    .padding({
      left: $r('app.string.lpx_32'),
      right: $r('app.string.lpx_32')
    })
  }
}
