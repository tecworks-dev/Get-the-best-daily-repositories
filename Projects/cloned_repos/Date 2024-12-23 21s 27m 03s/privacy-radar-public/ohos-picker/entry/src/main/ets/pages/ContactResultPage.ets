import { contact } from '@kit.ContactsKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { ToolBar } from '../widget/ToolBar';
import { Constants } from '../util/Constants';
import { lpx2fp } from '../util/DesignShow';
import App from '../app/App';

@Entry
@Component
struct ContactResultPage {
  @State contactList: Array<contact.Contact> = []

  aboutToAppear(): void {
    let promise = contact.selectContacts();
    promise.then((data) => {
      this.contactList = data
      console.log(`selectContact success: data->${JSON.stringify(data)}`);
      App.get().addContactList(this.contactList)
    }).catch((err: BusinessError) => {
      console.error(`selectContact fail: err->${JSON.stringify(err)}`);
    });
  }

  build() {
    Column() {
      ToolBar({
        title: '隐私雷达|联系人权限',
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
      Span(`获取联系人\n${this.contactList.length}`)
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
      ForEach(this.contactList,(item: contact.Contact,index: number) => {
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

  @Builder itemLayout(item: contact.Contact,index: number) {
    Column(){
      Text(`${item.name?.fullName ?? ''}`)
        .width(Constants.FULL_PARENT)
        .fontSize(lpx2fp(28))
        .fontColor($r('app.color.black'))
        .maxLines(1)
        .textOverflow({overflow: TextOverflow.Ellipsis})
      Text(`${item.phoneNumbers?.[0]?.phoneNumber ?? ''}`)
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