import promptAction from '@ohos.promptAction'
export class ToastUtils {
  show(message: string | Resource,duration?: number) {
    promptAction.showToast({message: message,duration: duration ? duration : 1500})
  }
}

export default new ToastUtils()