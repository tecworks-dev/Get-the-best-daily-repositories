export enum CommonNumbers {
  Zero = 0,
  One = 1,
  Two = 2,
  Three = 3,
  Four = 4,
  Five = 5,
  Six = 6,
  Seven = 7,
  Eight = 8,
  Nine = 9,
  Ten = 10
}

export enum GoodsType {
  Image = 0, // 图片
  Audio = 1, // 音频
  Video = 2, // 视频
  Column = 3, // 专栏
  SpecialColumn = 4, // 大专栏
  OfflineCourse = 5 // 线下课程
}

export enum VideoTrialStatus {
  TrialEnabled = 0, // 片段试看已开启
  TrialDisabled = 1 // 片段试看已关闭
}

export enum OnlineFlag {
  Online = 0, // 资源为线上状态
  Offline = 1 // 资源为线下状态
}

export enum SaleType {
  Free = 0, // 免费
  Paid = 1, // 付费
  Encrypted = 2, // 加密
  SpecifiedStudent = 3 // 指定学员
}

// 原生云播放器状态
export enum PlayerState {
  Idle = 'idle',
  Initialized = 'initialized',
  Prepared = 'prepared',
  Playing = 'playing',
  Paused = 'paused',
  Completed = 'completed',
  Stopped = 'stopped',
  Released = 'released',
  Error = 'error'
}

// 阿里云播放器状态
export enum AliPlayerState {
  Unknown = -1, // 未知
  Empty = 0, // 空状态
  Initializing = 1, // 初始化
  ReadySuccess = 2, // 准备成功
  Playing = 3, // 正在播放
  Paused = 4, // 已暂停
  Stopped = 5, // 已停止
  Completed = 6, // 播放完成
  Error = 7 // 播放出错
}

/**
 * 购买来源
 */
export enum BuySource {
  CourseDetail = 1, // 课程详情
  CourseList = 2 // 课程列表
}

/**
 * 支付方式
 */
export enum PaymentMethod {
  WECHAT_PAY = 1, // 微信支付
  ALIPAY = 2, // 支付宝支付
  HUAWEI_PAY = 3 // 华为支付
}

export enum SeekType {
  NORMAL = 0, // 普通seek
  ACCURATE = 1 // 精准seek
}