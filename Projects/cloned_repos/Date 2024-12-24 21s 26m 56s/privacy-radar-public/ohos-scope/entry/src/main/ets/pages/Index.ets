import App from '../app/App';
import { NavItem, NavList } from '../model/HomeTabModel';
import { Constants } from '../util/Constants';
import { lpx2fp } from '../util/DesignShow';
import PermissionHelper from '../util/PermissionHelper';
import { common } from '@kit.AbilityKit';
import PictureSelectUtil from '../util/PictureSelectUtil';
import RouterPath from '../util/RouterPath';
import picker from '@ohos.file.picker';
import { BusinessError } from '@kit.BasicServicesKit';
import CommonTipDialog from '../dialog/CommonTipDialog';
import DataUtil from '../util/DataUtil';
import { PermissionInfo } from '../data/PermissionInfo';

@Entry
@Component
struct Index {
  @State menuList: Array<PermissionInfo> = []
  @State message: string = '';
  @State tip: string = '欢迎来到隐私雷达\n请完成测试';
  @State testCompleted: boolean = false;

  @State albumResult: number = -1


  permissionTipDlgController: CustomDialogController = new CustomDialogController({
    builder: CommonTipDialog(
      {
        title: `${this.message}`,
        confirm: () => {
        }
      }),
    alignment: DialogAlignment.Center,
    offset: { dx: 0, dy: 0 },
    customStyle: true,
    autoCancel: false
  });

  onPageShow(): void {
    this.menuList = DataUtil.getMenuList()
    this.testCompleted = App.get().isTestCompleted()
    if (this.testCompleted) {
      this.tip = '欢迎来到隐私雷达\n测试已完成';
    } else {
      this.tip = '欢迎来到隐私雷达\n请完成测试';
    }
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
    Text('隐私雷达')
      .width(Constants.FULL_PARENT)
      .fontWeight(FontWeight.Bold)
      .fontSize(lpx2fp(36))
      .fontColor($r('app.color.black'))
      .margin({ top: $r('app.string.lpx_80') })
      .padding({ left: $r('app.string.lpx_32') })
    Text('by 林亦 LYi')
      .width(Constants.FULL_PARENT)
      .fontSize(lpx2fp(28))
      .fontColor($r('app.color.color_999999'))
      .margin({ top: $r('app.string.lpx_10') })
      .padding({ left: $r('app.string.lpx_32') })
  }

  @Builder content() {
    Text(this.tip)
      .width(Constants.FULL_PARENT)
      .fontColor($r('app.color.black'))
      .fontSize(lpx2fp(56))
      .textAlign(TextAlign.Center)
      .margin({
        top: $r('app.string.lpx_30'),
        left: $r('app.string.lpx_30')
      })

    if (this.testCompleted) {
      Text("查看报告")
        .height($r('app.string.lpx_80'))
        .margin({
          top: $r('app.string.lpx_30'),
          bottom: $r('app.string.lpx_20')
        })
        .padding({
          left: $r('app.string.lpx_60'),
          right: $r('app.string.lpx_60')
        })
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(32))
        .backgroundColor($r('app.color.color_EAEAEA'))
        .borderRadius($r('app.string.lpx_40'))
        .onClick(() => {
          App.get().pushUrl({
            url: RouterPath.TEST_REPORT_PAGE
          })
        })
    }

    List({ space: 10 }) {
      ForEach(this.menuList,(item: PermissionInfo,index: number) => {
        ListItem() {
          this.itemLayout(item,index)
        }
      })
    }
    .width(Constants.FULL_PARENT)
    .layoutWeight(1)
    .lanes(2)
    .margin({
      top: $r('app.string.lpx_30')
    })
    .padding($r('app.string.lpx_10'))
  }

  @Builder itemLayout(item: PermissionInfo,index: number) {
    Column(){
      Image(item.icon)
        .width($r('app.string.lpx_80'))
        .height($r('app.string.lpx_80'))

      Text(item.content)
        .fontColor($r('app.color.black'))
        .fontSize(lpx2fp(32))
        .margin({top: $r('app.string.lpx_20')})
      Text(item.status)
        .fontColor($r('app.color.color_999999'))
        .fontSize(lpx2fp(24))
    }
    .width("95%")
    .height($r('app.string.lpx_300'))
    .backgroundColor($r('app.color.color_EAEAEA'))
    .justifyContent(FlexAlign.Center)
    .onClick(() => {
      const context = getContext(this) as common.UIAbilityContext;
      if (index === 0) {//相册
        // PermissionHelper.checkPermissions(context, 'ohos.permission.READ_MEDIA').then((grant: boolean) => {
        //   if (grant) {
        //     //在添加上传按钮之前上传图片避免重新处理数据
        //     PictureSelectUtil.fileSelect(10).then((value) => {
        //       if (value.length > 0) {
        //         App.get().pushUrl({
        //           url: RouterPath.ALBUM_RESULT_PAGE,
        //           params: {
        //             data: value
        //           }
        //         })
        //       }
        //     })
        //   } else {
        //     PermissionHelper.openPermissionsInSystemSettings(context)
        //   }
        // })
        App.get().pushUrl({
          url: RouterPath.ALBUM_RESULT_PAGE
        })
      } else if (index === 1) {//联系人(模拟器跑不起来需要真机验证)
        //android 获取联系人权限：
        // EasyPermissions.requestPermissions(
        //   this,
        //   getString(R.string.permission_contacts_rationale_message),
        //   REQUEST_CODE_CONTACTS_PERMISSION,
        //   Manifest.permission.READ_CONTACTS
        // )
        App.get().pushUrl({
          url: RouterPath.CONTACT_RESULT_PAGE
        })
      } else if (index === 2) {//文件权限
        const documentSelectOptions = new picker.DocumentSelectOptions()
        documentSelectOptions.maxSelectNumber = 5; // 选择文档的最大数目（可选）
        // documentSelectOptions.defaultFilePathUri = "file://docs/storage/Users/currentUser/test"; // 指定选择的文件或者目录路径（可选）
        documentSelectOptions.fileSuffixFilters = ['.webp','.gif','.png','.jpeg','.jpg', '.txt', '.mp4','.mp3','.doc','.docx','.pdf','.txt','.ppt','.pptx','.xls','.xlsx']; // 选择文件的后缀类型，若选择项存在多个后缀名，则每一个后缀名之间用英文逗号进行分隔（可选）

        let uris: Array<string> = [];
        const documentViewPicker = new picker.DocumentViewPicker(); // 创建文件选择器实例
        documentViewPicker.select(documentSelectOptions).then((documentSelectResult: Array<string>) => {
          uris = documentSelectResult;
          console.info('documentViewPicker.select to file succeed and uris are:' + uris);
          App.get().pushUrl({
            url: RouterPath.FILE_RESULT_PAGE,
            params: {
              data: documentSelectResult
            }
          })
        }).catch((err: BusinessError) => {
          console.error(`Invoke documentViewPicker.select failed, code is ${err.code}, message is ${err.message}`);
        })
      } else if (index === 3) {//相机权限
         PermissionHelper.checkPermissions(context, 'ohos.permission.CAMERA').then((grant: boolean) => {
              if (grant) {
                App.get().pushUrl({
                  url: RouterPath.CAPTURE_PAGE
                })
              } else {
                PermissionHelper.openPermissionsInSystemSettings(context)
              }
            })
      } else if (index === 4) {//手机信息，需要权限获取唯一标识
        App.get().pushUrl({
          url: RouterPath.DEVICE_RESULT_PAGE
        })
      } else if (index === 5) {//通话记录
        // android 获取通话记录权限：
        // EasyPermissions.requestPermissions(
        //   this,
        //   getString(R.string.permission_read_call_log_rationale_message),
        //   REQUEST_CODE_READ_CALL_LOG_PERMISSION,
        //   Manifest.permission.READ_CALL_LOG
        // )
        //鸿蒙通话记录权限是首先开放的，需要准备材料在应用市场申请。默认是不开放的。文档参考：
        //https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V3/security-permissions-available-0000001051089272-V3
        App.get().pushUrl({
          url: RouterPath.SECURE_RESULT_PAGE,
          params: {
            data: '获取通话记录权限',
            title: '隐私雷达|通话记录权限',
            type: 1
          }
        })
      } else if (index === 6) {//短信
    //     android 获取短信权限：
    //     EasyPermissions.requestPermissions(
    //       this,
    //       getString(R.string.permission_read_sms_rationale_message),
    //       REQUEST_CODE_READ_SMS_PERMISSION,
    //       Manifest.permission.READ_SMS
    //     )
        //鸿蒙短信权限是首先开放的，需要准备材料在应用市场申请。默认是不开放的。文档参考：
        //https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V3/security-permissions-available-0000001051089272-V3
        App.get().pushUrl({
          url: RouterPath.SECURE_RESULT_PAGE,
          params: {
            data: '获取短信权限',
            title: '隐私雷达|短信权限',
            type: 2
          }
        })
      } else if (index === 7) {//应用列表
        //鸿蒙next不允许查询设备应用列表数据，相关权限无法申请，官方文档：
        //https://developer.huawei.com/consumer/cn/doc/harmonyos-faqs-V5/faqs-ability-104-V5
        App.get().pushUrl({
          url: RouterPath.SECURE_RESULT_PAGE,
          params: {
            data: '获取应用列表权限',
            title: '隐私雷达|应用列表权限',
            type: 3
          }
        })
      }
    })

  }
}