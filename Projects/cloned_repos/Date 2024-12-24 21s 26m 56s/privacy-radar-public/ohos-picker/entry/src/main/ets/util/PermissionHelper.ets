import { abilityAccessCtrl, bundleManager, common, Permissions, Want } from '@kit.AbilityKit';
import { BusinessError } from '@kit.BasicServicesKit';

class PermissionHelper {

  async checkPermissionGrant(permission: Permissions): Promise<boolean> {
    let atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
    let grantStatus: abilityAccessCtrl.GrantStatus = abilityAccessCtrl.GrantStatus.PERMISSION_DENIED;

    // 获取应用程序的accessTokenID
    let tokenId: number = 0;
    try {
      let bundleInfo: bundleManager.BundleInfo = await bundleManager.getBundleInfoForSelf(bundleManager.BundleFlag.GET_BUNDLE_INFO_WITH_APPLICATION);
      let appInfo: bundleManager.ApplicationInfo = bundleInfo.appInfo;
      tokenId = appInfo.accessTokenId;
    } catch (error) {
      const err: BusinessError = error as BusinessError;
      console.error(`Failed to get bundle info for self. Code is ${err.code}, message is ${err.message}`);
    }

    // 校验应用是否被授予权限
    try {
      grantStatus = await atManager.checkAccessToken(tokenId, permission);
    } catch (error) {
      const err: BusinessError = error as BusinessError;
      console.error(`Failed to check access token. Code is ${err.code}, message is ${err.message}`);
    }

    return grantStatus === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED;
  }

  async checkPermissions(context: common.UIAbilityContext,permissions: Permissions): Promise<boolean> {
    let grantStatus: boolean = await this.checkPermissionGrant(permissions);
    return new Promise(async (resolve: Function) => {
      if (grantStatus) {
        resolve(true)
      } else {
        resolve(this.reqPermissionsFromUser([permissions],context))
      }
    })
  }
  async hasPermissions(permissions: Permissions): Promise<boolean> {
    let grantStatus: boolean = await this.checkPermissionGrant(permissions);
    return new Promise(async (resolve: Function) => {
      if (grantStatus) {
        resolve(true)
      } else {
        resolve(false)
      }
    })
  }

  // 使用UIExtensionAbility：将common.UIAbilityContext 替换为common.UIExtensionContext
  reqPermissionsFromUser(permissions: Array<Permissions>, context: common.UIAbilityContext): Promise<boolean> {
    return new Promise(async (resolve: Function) => {
      let atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
      // requestPermissionsFromUser会判断权限的授权状态来决定是否唤起弹窗
      atManager.requestPermissionsFromUser(context, permissions).then((data) => {
        let grantStatus: Array<number> = data.authResults;
        let length: number = grantStatus.length;
        if (this.permissionGrantCheck(data.authResults)) {
          // 用户授权，可以继续访问目标操作
          resolve(true)
        } else {
          // 用户拒绝授权，提示用户必须授权才能访问当前页面的功能，并引导用户到系统设置中打开相应的权限
          resolve(false)
        }
        // 授权成功
      }).catch((err: BusinessError) => {
        console.error(`Failed to request permissions from user. Code is ${err.code}, message is ${err.message}`);
        return false;
      })
    })
  }

  /**
   * 权限返回数组检查
   * 由于data.authResults.every((value) => value === 0)永远返回的是false
   * 无论用户是否全部赋予权限或者不赋予权限，但实际data.authResults中的值与用户的操作预期是一致的
   * 所以添加了一个方法去除掉这种错误
   * @param resultList requestPermissionsFromUser方法返回的用户是否同意赋予权限
   * @returns
   */
  permissionGrantCheck(resultList:Array<number>):boolean {
    if (resultList.length<=0) {
      return false
    }
    const numbers = new Array<number>();
    for (let i = 0; i < resultList.length; i++) {
      numbers.push(resultList[i])
    }
    const result = numbers.every((value) => value === 0)
    return result;
  }

  // 使用UIExtensionAbility：将common.UIAbilityContext 替换为common.UIExtensionContext
  openPermissionsInSystemSettings(context: common.UIAbilityContext): void {
    let wantInfo: Want = {
      bundleName: 'com.huawei.hmos.settings',
      abilityName: 'com.huawei.hmos.settings.MainAbility',
      uri: 'application_info_entry',
      parameters: {
        pushParams: 'com.lyihub.privacy_radar' // 打开指定应用的详情页面
      }
    }
    context.startAbility(wantInfo).then(() => {
      console.error('startAbility success')
    }).catch((err: BusinessError) => {
      console.error('startAbility failed', JSON.stringify(err))
    })
  }
}

export default new PermissionHelper()