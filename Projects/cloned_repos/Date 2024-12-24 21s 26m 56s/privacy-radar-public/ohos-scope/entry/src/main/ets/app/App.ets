import HashMap from '@ohos.util.HashMap';
import router from '@ohos.router';
import Logger from '../util/Logger';
import RouterPath from '../util/RouterPath';
import { Constants } from '../util/Constants';
import { TestReportInfo } from '../data/TestReportInfo';
import { contact } from '@kit.ContactsKit';

export class App {
  private TAG = 'App'
  private static instance: App
  private routerLaunchMap = new HashMap<string,string>()
  private testMap = new HashMap<string,number>()
  // private albumList: Array<string> = []
  // private contactList: Array<contact.Contact> = []
  private fileList: Array<string> = []

  private constructor() {
    // 私有构造函数，防止外部实例化
  }

  public static get(): App {
    if (!App.instance) {
      App.instance = new App()
    }
    return App.instance
  }
  
  getTestResult(info: TestReportInfo): string {
    if (info.resultType == 0) {
      return `${info.result}${info.resultUnit}`
    }
    return '成功'
    
  }

  getTestReport(): Array<TestReportInfo> {
    let reportList = Array<TestReportInfo>()

    let albumCount = this.getAlbumCount()
    let contactCount = this.getContactCount()
    let fileCount = this.getFileCount()
    let scanResult = this.getScanResult()
    let deviceResult = this.getDeviceResult()
    // let callLogCount = this.getCallLogCount()
    // let smsCount = this.getSmsCount()
    // let installAppCount = this.getAppCount()

    let item1 = new TestReportInfo()
    item1.resultType = 0
    item1.content = "已获取相册照片"
    item1.result = albumCount
    item1.resultUnit = "张"

    let item2 = new TestReportInfo()
    item2.resultType = 0
    item2.content = "已获取联系人"
    item2.result = contactCount
    item2.resultUnit = "个"

    let item3 = new TestReportInfo()
    item3.resultType = 0
    item3.content = "已获取文件"
    item3.result = fileCount
    item3.resultUnit = "个"

    let item4 = new TestReportInfo()
    item4.resultType = 1
    item4.content = "二维码扫描结果"
    item4.result = scanResult

    let item5 = new TestReportInfo()
    item5.resultType = 1
    item5.content = "手机信息"
    item5.result = deviceResult

    let item6 = new TestReportInfo()
    item6.resultType = 0
    item6.content = "已获取通话记录"
    item6.result = 0
    item6.resultUnit = "条"

    let item7 = new TestReportInfo()
    item7.resultType = 0
    item7.content = "已获取短信"
    item7.result = 0
    item7.resultUnit = "条"

    let item8 = new TestReportInfo()
    item8.resultType = 0
    item8.content = "获取已安装app"
    item8.result = 0
    item8.resultUnit = "个"

    reportList.push(item1)
    reportList.push(item2)
    reportList.push(item3)
    reportList.push(item4)
    reportList.push(item5)
    reportList.push(item6)
    reportList.push(item7)
    reportList.push(item8)

    return reportList
  }

  addAlbumList(count: number ) {
    // this.albumList = this.albumList.concat(albumInfos)
    this.putAlbumCount(count)
  }

  addContactList(count: number ) {
    // this.contactList = this.contactList.concat(albumInfos)
    this.putContactCount(count)
  }

  addFileList(fileInfos: Array<string> ) {
    this.fileList = this.fileList.concat(fileInfos)
    this.putFileCount(this.fileList.length)
  }

  putAlbumCount(count: number) {
    this.testMap.set(Constants.ALBUM_COUNT_KEY,count)
  }

  getAlbumCount():number {
    return this.testMap.get(Constants.ALBUM_COUNT_KEY) ?? -1
  }

  putContactCount(count: number) {
    this.testMap.set(Constants.CONTACT_COUNT_KEY,count)
  }

  getContactCount():number {
    return this.testMap.get(Constants.CONTACT_COUNT_KEY) ?? -1
  }

  putFileCount(count: number) {
    this.testMap.set(Constants.FILE_COUNT_KEY,count)
  }

  getFileCount():number {
    return this.testMap.get(Constants.FILE_COUNT_KEY) ?? -1
  }

  putScanResult(count: number) {
    this.testMap.set(Constants.SCAN_RESULT_KEY,count)
  }

  getScanResult():number {
    return this.testMap.get(Constants.SCAN_RESULT_KEY) ?? -1
  }

  putDeviceResult(count: number) {
    this.testMap.set(Constants.DEVICE_RESULT_KEY,count)
  }

  getDeviceResult():number {
    return this.testMap.get(Constants.DEVICE_RESULT_KEY) ?? -1
  }

  putCallLogCount(count: number) {
    this.testMap.set(Constants.CALL_LOG_KEY,count)
  }
  getCallLogCount():number {
    return this.testMap.get(Constants.CALL_LOG_KEY) ?? -1
  }

  putSmsCount(count: number) {
    this.testMap.set(Constants.SMS_KEY,count)
  }
  getSmsCount():number {
    return this.testMap.get(Constants.SMS_KEY) ?? -1
  }

  putAppCount(count: number) {
    this.testMap.set(Constants.APP_LIST_KEY,count)
  }
  getAppCount():number {
    return this.testMap.get(Constants.APP_LIST_KEY) ?? -1
  }

  isTestCompleted(): boolean {
    let compeleted:boolean = false
    let albumCount = this.getAlbumCount()
    let contactCount = this.getContactCount()
    let fileCount = this.getFileCount()
    let scanResult = this.getScanResult()
    let deviceResult = this.getDeviceResult()
    // let callLogCount = this.getCallLogCount()
    // let smsCount = this.getSmsCount()
    // let installAppCount = this.getAppCount()

    compeleted = albumCount != -1 && contactCount != -1 && fileCount != -1 && scanResult != -1 && deviceResult != -1

    return compeleted
  }

  pushUrl(options: router.RouterOptions, mode?: router.RouterMode) {
    Logger.error(this.TAG,`pushUrl-url = ${options.url}`)
    this.routerLaunchMap.set(options.url,options.url)
    router.pushUrl(options,mode ? mode : router.RouterMode.Standard).catch((error ? : '') => {
      Logger.error(this.TAG, `pushUrl error = ${JSON.stringify(error)}`);
    });

    this.logLaunchRouter()
  }

  replaceUrl(options: router.RouterOptions, mode?: router.RouterMode) {
    Logger.error(this.TAG,`replaceUrl-url = ${options.url}`)

    this.routerLaunchMap.set(options.url,options.url)

    router.replaceUrl(options,mode ? mode : router.RouterMode.Standard).catch((error ? : '') => {
      Logger.error(this.TAG, `replaceUrl error = ${JSON.stringify(error)}`);
    });

    this.logLaunchRouter()
  }

  back() {
    let path = router.getState().path
    let name = router.getState().name
    let url = `${path}${name}`
    this.routerLaunchMap.remove(url)

    router.back()
    Logger.error(this.TAG,`back-url = ${url}`)
    this.logLaunchRouter()
  }

  backUrl(url: string) {
    this.routerLaunchMap.remove(url)
    router.back({
      url: url
    })
    Logger.error(this.TAG,`backRouter-url = ${url}`)
    this.logLaunchRouter()
  }

  backOtherThenMainPage() {
    this.routerLaunchMap.forEach((value, key: string) => {
      Logger.error(this.TAG,`backOtherThenMainPage-key = ${key}`)
      if (key != RouterPath.MAIN_PAGE) {
        this.backUrl(key)
      }
    })
  }

  logLaunchRouter() {
    Logger.error(this.TAG,`logLaunchRouter-length = ${this.routerLaunchMap.length}`)
    this.routerLaunchMap.forEach((value, key) => {
      Logger.error(this.TAG,`logLaunchRouter-key = ${key}`)
      Logger.error(this.TAG,`logLaunchRouter-value = ${value}`)
    });
  }
}

export default App