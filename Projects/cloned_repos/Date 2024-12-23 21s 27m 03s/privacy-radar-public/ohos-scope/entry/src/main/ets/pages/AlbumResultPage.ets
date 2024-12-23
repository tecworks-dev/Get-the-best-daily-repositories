import { Constants } from '../util/Constants';
import { router } from '@kit.ArkUI';
import { ToolBar } from '../widget/ToolBar';
import { lpx2fp } from '../util/DesignShow';
import App from '../app/App';

// 20241114_UPDATE: Use restricted permission to get photos
import dataSharePredicates from '@ohos.data.dataSharePredicates';
import photoAccessHelper from '@ohos.file.photoAccessHelper';
import { abilityAccessCtrl, Permissions } from '@kit.AbilityKit';

const context = getContext(this);
let phAccessHelper = photoAccessHelper.getPhotoAccessHelper(context);

const ALBUM_DISPLAY_LIMIT = 30;
const TOTAL_DISPLAY_LIMIT = 10000;
// let photoURLs: Array<string> = [];

@Entry
@Component
struct AlbumResultPage {
  @State albumList: Array<string> = [];
  @State photoList: Array<PixelMap> = [];
  @State isScanning: boolean = false;
  @State total: number = 0;

  // Obtain all photos in system album
  async getAllPhotosByAlbumType(albumType: photoAccessHelper.AlbumType = photoAccessHelper.AlbumType.SYSTEM) {
    // Scanning Status
    this.isScanning = true;

    const predicates: dataSharePredicates.DataSharePredicates = new dataSharePredicates.DataSharePredicates();
    const fetchOptions: photoAccessHelper.FetchOptions = {
      fetchColumns: [],
      predicates: predicates
    };

    try {
      // Reset properties
      this.total = 0;
      const tmpImageList: PixelMap[] = [];
      // photoURLs = [];

      // Get images
      const albumFetchResult: photoAccessHelper.FetchResult<photoAccessHelper.Album> = await phAccessHelper.getAlbums(albumType, photoAccessHelper.AlbumSubtype.ANY);
      albumFetchResult.getAllObjects(async (err, photoAssetsList) => {
        photoAssetsList.forEach(async (album, albumIndex) => {
          // let albumFetchResult: photoAccessHelper.FetchResult<photoAccessHelper.PhotoAsset> = await album.getAssets(fetchOptions);
          // const list = await albumFetchResult.getAllObjects();

          // Fetch photo assets
          const photoFetchResult: photoAccessHelper.FetchResult<photoAccessHelper.PhotoAsset> = await album.getAssets(fetchOptions);
          const list = await photoFetchResult.getAllObjects(); // tmp list

          // Display ALL Image
          for(let idx = 0; idx < list.length; idx++) {

            // still count when reach TOTAL DISPLAY COUNT(10k), but will not display
            this.total++;
            App.get().addAlbumList(this.total);

            if(this.total >= TOTAL_DISPLAY_LIMIT) return;

            const el = list[idx];
            const thm:PixelMap = await el.getThumbnail();

            // Add the thumbnail pixel map to the data list, provided the limit isn't reached.
            if(idx < ALBUM_DISPLAY_LIMIT) {
              tmpImageList.push(thm);
            }
            // Update view
            if (idx === TOTAL_DISPLAY_LIMIT || idx === list.length - 1) {
              this.photoList.push(...tmpImageList);
            }
          }

          // Write into result
          if(albumIndex === photoAssetsList.length - 1){
            // App.get().addAlbumList(this.total);
            this.isScanning = false;
          }

          // Close fetch session
          photoFetchResult.close();

        });


      });
      albumFetchResult.close();

    } catch (err) {
      console.error('favorite failed with err: ' + err);
    }

  }

  aboutToAppear(): void {
    const permissions: Array<Permissions> = ['ohos.permission.READ_IMAGEVIDEO'];

    // Make permission request and response accordingly
    abilityAccessCtrl.createAtManager().requestPermissionsFromUser(context, permissions).then(async () => {
      try {
        this.photoList = [];
        await this.getAllPhotosByAlbumType(photoAccessHelper.AlbumType.SYSTEM);
        // await this.getAllPhotosByAlbumType(photoAccessHelper.AlbumType.USER);
      } catch(err) {
        // Do nothing
      }
    });

    // this.albumList = (router.getParams() as object)?.[Constants.DATA]

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
      Span(this.isScanning ? `正在扫描\n${this.total}` : `已成功获取\n${this.total}`)
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

    // Column(){
    //   Image(this.photoList[0])
    //     .width(Constants.FULL_PARENT)
    //     .height(Constants.FULL_PARENT)
    // }
    // .width("95%")
    // .height($r('app.string.lpx_250'))
    // .backgroundColor($r('app.color.color_33EB4F3A'))
    // .justifyContent(FlexAlign.Center)

    List({ space: 10 }) {
      ForEach(this.photoList,(item: string,index: number) => {
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

    Text(`每相册取 ${ALBUM_DISPLAY_LIMIT} 张, 最多 ${TOTAL_DISPLAY_LIMIT} 张`)
      .fontSize(10)
      .opacity(0.5)
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