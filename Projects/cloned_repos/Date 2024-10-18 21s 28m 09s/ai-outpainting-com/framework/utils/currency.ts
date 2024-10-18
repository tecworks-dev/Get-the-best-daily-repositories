// 元转分
export function yuanToFen(yuan: number) {
  if (isNaN(yuan)) {
    throw new Error('输入值必须是一个数字')
  }
  return Math.round(yuan * 100)
}

// 分转元，并保留两位小数
export function fenToYuan(fen: number) {
  if (isNaN(fen)) {
    throw new Error('输入值必须是一个数字')
  }
  return (fen / 100).toFixed(2)
}
