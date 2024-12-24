import { display, router } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { scanCore, scanBarcode, customScan } from '@kit.ScanKit';
import { BusinessError } from '@kit.BasicServicesKit';
import App from '../app/App';
import RouterPath from '../util/RouterPath';
import { lpx2fp } from '../util/DesignShow';

const TAG: string = 'ScanKit customScanPage';
let setInterFlag: number;
let toResultPage: number;

@Extend(Button)
function button() {
  .backgroundColor($r('sys.color.ohos_id_color_text_primary_activated'))
  .fontColor(Color.White)
  .align(Alignment.Center)
  .type(ButtonType.Capsule)
  .margin({ bottom: 10 })
  .width(200)
  .height(40)
  .borderRadius(20)
}

@Entry
@Component
struct CapturePage {
  @State userGrant: boolean = false // Whether the camera permission has been requested.
  @State showReScan: boolean = false // Whether the barcode has been detected in the image.

  @State scanCodeRectCallback: Array<scanBarcode.ScanCodeRect> = [] // Position of the barcode image, returned in a callback. The position type is scanBarcode.ScanCodeRect.
  @State scanCodeRect: Array<scanBarcode.ScanResult> = []
  @State scanCodeRectLength: number = 0
  @State scanCodeRectFrameCallback: Array<scanBarcode.ScanCodeRect> = [] // Position of the barcode image in YUV data, returned in frameCallback. The position type is scanBarcode.ScanCodeRect.
  @State scanLineY: string = '0'

  aboutToAppear(): void {
    // do nothing
  }

  // Display the page.
  async onPageShow() {
    hilog.error(0x0001, TAG, `onPageShow - requestCameraPermission before this.userGrant: ${this.userGrant}`);
    // Check whether the device is foldable.
    if (display.isFoldable()) {
      this.foldStatusSubscribe();
    }
    this.startQRScan();
  }

  // Hide the page.
  async onPageHide() {
    this.userGrant = false;
    clearTimeout(setInterFlag);
    // Stop and release camera streams when the scanning UI disappears or is hidden.
  }

  aboutToDisappear(): void {
    if (display.isFoldable()) {
      // Cancel the foldable screen status change listener.
      display.off('foldStatusChange');
    }
  }

  // Customize the back button and scanning prompt on the top of the scanning UI.
  @Builder
  TopTool() {
    Column() {
      Flex({ direction: FlexDirection.Row, justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
        Image($r('app.media.ic_scan_close'))
          .fillColor(Color.Black)
          .width(30)
          .height(30)
          .objectFit(ImageFit.Contain)
          .onClick(async () => {
            router.back();
          })
      }.padding({ left: 24, right: 24, top: 24 })

      Column() {
        Text($r('app.string.Scan_barcode'))
          .fontColor($r('app.color.white'))
          .fontSize(lpx2fp(28))
        Text($r('app.string.Scan_barcode_tips'))
          .fontColor($r('app.color.white'))
          .fontSize(lpx2fp(28))
      }.margin({ left: 24, right: 24, top: 16 })
    }
    .height(146)
    .width('100%')
    .margin({ top: 24 })
  }

  build() {

    Stack() {

      // Customize the layout.
      Column() {
        this.TopTool()
        Column() {
        }
        .layoutWeight(1)
        .width('100%')

        Column() {
          if(this.showReScan){
            // Rescan button.
            Button('Scan').button()
              .onClick(() => {
                // Reset scanning options.
                this.scanCodeRectCallback = [];
                this.scanCodeRectFrameCallback = [];
                this.showReScan = false;
                this.scanCodeRectLength = 0;
                this.scanLineY = '0%';
                clearTimeout(toResultPage);
                // this.customScanInit();
                this.startQRScan();
              })
          } else {
            Text("读取中...").fontSize(16)
          }
        }
        .height(250)
      }
      .width('100%')
      .height('100%')

    }
    // You are advised to set camera streams to the full screen mode.
    .width('100%')
    .height('100%')
  }

  // Scanning result returned in a callback.
  async showScanResult(result: Array<scanBarcode.ScanResult>) {
    this.scanCodeRectLength = result.length;
    if (result.length > 0) {
      clearTimeout(toResultPage);
      // Position of the barcode image, which is of the type scanBarcode.ScanCodeRect.
      result.forEach((item: scanBarcode.ScanResult) => {
        if (item.scanCodeRect) {
          this.scanCodeRectCallback.push(item.scanCodeRect);
        }
      })

      console.warn("CP RES:" + JSON.stringify(result));
      // Stop displaying the message indicating that no barcode is recognized after the barcode is actually recognized.
      clearTimeout(setInterFlag);
      // Print the test result. You can process the obtained result accordingly.
      hilog.info(0x0001, TAG, `Succeeded in getting ScanResult.`);
      if (this.scanCodeRectLength === 1) {
        toResultPage = setTimeout(() => {
          App.get().replaceUrl({
            url: RouterPath.CAPTURE_RESULT_PAGE,
            params : {
              result: result
            }
          })
        }, 1000);
      }
    }
  }

  startQRScan(): void {
    let options: scanBarcode.ScanOptions = {
      scanTypes: [scanCore.ScanType.ALL],
      enableMultiMode: true, // Whether to enable the function of scanning multiple barcodes.
      enableAlbum: true // Whether to display the gallery button.
    }
    try {
      // 可调用getContext接口获取当前页面关联的UIAbilityContext
      scanBarcode.startScanForResult(getContext(this), options,
        (error: BusinessError, result: scanBarcode.ScanResult) => {

          if (error) {
            this.showReScan = true;
            hilog.error(0x0001, '[QRScan]',
              `Failed to get ScanResult by callback with options. Code: ${error.code}, message: ${error.message}`);
            return;
          }

          this.showReScan = false;
          // Display Scan Reulst
          this.showScanResult([result]);

          // 解析码值结果跳转应用服务页
          hilog.info(0x0001, '[QRScan]', `Succeeded in getting ScanResult by callback with options, result is ${JSON.stringify(result)}`);
        })
    } catch (error) {
      hilog.error(0x0001, '[Scan CPSample]',
        `Failed to start the scanning service. Code:${error.code}, message: ${error.message}`);
    }
  }

  // Listen to the status change of the foldable screen, and release and restart of camera streams.
  foldStatusSubscribe(): void {
    try {
      display.on('foldStatusChange', async (foldStatus: display.FoldStatus) => {
        // 1: expanded; 2: collapsed. When the status changes, the width and height need to be calculated again.
        if (foldStatus === display.FoldStatus.FOLD_STATUS_EXPANDED ||
          foldStatus === display.FoldStatus.FOLD_STATUS_FOLDED) {
          this.scanCodeRectCallback = [];
          this.scanCodeRectFrameCallback = [];
          clearTimeout(toResultPage);
          clearTimeout(setInterFlag);
          // Restart camera streams.
          setTimeout(() => {
            this.startQRScan();
          }, 300);
        }
      })
    } catch (error) {
      hilog.error(0x0001, TAG, `Failed to foldStatusChange. Code: ${error.code}, message: ${error.message}`);
    }
  }
}