import { router } from '@kit.ArkUI';
import { scanBarcode } from '@kit.ScanKit';
import App from '../app/App';

@Extend(Text)
function textName() {
  .fontSize(20)
  .fontColor('#000000')
  .fontWeight(FontWeight.Medium)
  .width('100%')
  .margin({ bottom: 5 })
}

@Extend(Text)
function textValue() {
  .fontSize(16)
  .fontWeight(FontWeight.Normal)
  .width('100%')
  .margin({ bottom: 5 })
}

let typeStr: Array<string> =
  ['FORMAT_UNKNOWN', 'AZTEC_CODE', 'CODABAR_CODE', 'CODE39_CODE', 'CODE93_CODE', 'CODE128_CODE', 'DATAMATRIX_CODE',
    'EAN8_CODE', 'EAN13_CODE',
    'ITF14_CODE', 'PDF417_CODE', 'QR_CODE', 'UPC_A_CODE', 'UPC_E_CODE', ''];

@Entry
@Component
struct CaptureResultPage {
  private result: Array<scanBarcode.ScanResult> = []
  public scroller: Scroller = new Scroller()

  aboutToAppear() {
    let params = router.getParams() as Record<string, string | Array<scanBarcode.ScanResult>>;
    this.result = params.result as Array<scanBarcode.ScanResult>;
    App.get().putScanResult(this.result.length)
  }

  @Builder
  CustomTextBox(scanType: string, inputValue: string, left: number, top: number, right: number, bottom: number) {
    Column() {
      Column() {
        Text($r('app.string.code_format')).textName()
        Text(scanType).textValue()

      }
      .align(Alignment.Start)
      .margin({ left: 10, top: 20, bottom: 20 })

      Column() {
        Text($r('app.string.result_content')).textName()
          .fontWeight(FontWeight.Medium)
        Text(inputValue).textValue()
      }
      .align(Alignment.Start)
      .margin({ left: 10, bottom: 20 })

      Column() {
        Text($r('app.string.code_react')).textName()
        Text('left: ' + left + '  top: ' + top + '  right: ' + right + '  bottom: ' + bottom).textValue()
      }
      .align(Alignment.Start)
      .margin({ left: 10, bottom: 20 })
    }.margin({ bottom: 5 })
    .backgroundColor('#fffaf8f8')
    .borderRadius(8)
  }

  build() {
    Column() {
      Row() {
        Image($rawfile('scan_back.svg'))
          .fillColor(Color.Black)
          .width(30)
          .height(30)
          .objectFit(ImageFit.Contain)
        Text($r('app.string.title_result'))
          .fontSize(20)
          .fontColor('#000000')
          .margin({ left: 15 })
      }
      .margin({ top: 20 })
      .zIndex(10)
      .position({ x: 15, y: 0 })
      .onClick(async () => {
        router.back();
      })

      Scroll(this.scroller) {
        Column() {
          Column() {
            ForEach(this.result, (item: scanBarcode.ScanResult) => {
              if (item['scanCodeRect']) {
                this.CustomTextBox(typeStr[item['scanType']], item['originalValue'], item['scanCodeRect']['left'],
                  item['scanCodeRect'].top, item['scanCodeRect'].right, item['scanCodeRect'].bottom)
              }
            })
          }
        }
        .width(330)
      }
      .scrollable(ScrollDirection.Vertical)
      .scrollBar(BarState.On)
      .scrollBarColor(Color.Gray)
      .scrollBarWidth(5)
      .margin({ top: 120 })
    }
    .width('100%')
    .padding({
      top: 40
    })
  }
}