// PermissionsUtil.ets
import { bundleManager, common, abilityAccessCtrl, PermissionRequestResult, Permissions } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

const TAG: string = 'ScanKit Permission';
let context = getContext(this) as common.UIAbilityContext;

export class PermissionsUtil {
  public static async checkAccessToken(permission: Permissions): Promise<abilityAccessCtrl.GrantStatus> {
    let atManager = abilityAccessCtrl.createAtManager();
    let grantStatus: abilityAccessCtrl.GrantStatus = -1;
    // Obtain accessTokenID of the app.
    let tokenId: number = 0;
    try {
      let bundleInfo: bundleManager.BundleInfo = await bundleManager.getBundleInfoForSelf(bundleManager.BundleFlag.GET_BUNDLE_INFO_WITH_APPLICATION);
      let appInfo: bundleManager.ApplicationInfo = bundleInfo.appInfo;
      tokenId = appInfo.accessTokenId;
    } catch (error) {
      hilog.error(0x0001, TAG, 'Failed to get bundle info for self. error: %{public}s', JSON.stringify(error));
    }
    // Check whether the app has been granted the required permission.
    try {
      grantStatus = await atManager.checkAccessToken(tokenId, permission);
    } catch (error) {
      hilog.error(0x0001, TAG, 'Failed to check access token. error: %{public}s', JSON.stringify(error));
    }
    return grantStatus;
  }

  public static async reqPermissionsFromUser(): Promise<number[]> {
    hilog.info(0x0001, TAG, 'reqPermissionsFromUser start')
    let atManager = abilityAccessCtrl.createAtManager();
    let grantStatus: PermissionRequestResult = { permissions: [], authResults: [] }
    try {
      grantStatus = await atManager.requestPermissionsFromUser(context, ['ohos.permission.CAMERA']);
    } catch (error) {
      hilog.error(0x0001, TAG, 'Failed to check access token. error: %{public}s', JSON.stringify(error));
    }
    return grantStatus.authResults;
  }
}
