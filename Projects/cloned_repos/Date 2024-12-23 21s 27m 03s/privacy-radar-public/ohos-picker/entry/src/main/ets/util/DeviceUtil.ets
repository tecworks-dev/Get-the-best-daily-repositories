import deviceinfo from '@ohos.deviceInfo';

class DeviceUtil {

  getUDID(): string {
    let serial = deviceinfo.udid
    if (!serial) {
      serial = 'unknown'
    }
    return serial
  }

  getSerial(): string {
    let serial = deviceinfo.serial
    if (!serial) {
      serial = 'unknown'
    }
    return serial
  }

}

export default new DeviceUtil()