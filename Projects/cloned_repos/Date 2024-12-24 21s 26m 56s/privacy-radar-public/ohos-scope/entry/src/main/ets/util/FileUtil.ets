class FileUtil {
  getFileNameByPath(path: string): string {
    let fileName = ''
    const items = path.split('/')
    const count = items.length ?? 0
    if (count > 0) {
      fileName = items[items.length - 1]
    }
    return fileName
  }
}

export default new FileUtil()