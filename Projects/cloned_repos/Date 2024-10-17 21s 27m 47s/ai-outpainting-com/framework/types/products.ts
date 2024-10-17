/**
 * 产品
 */
export interface Product extends Record<string, any> {
  /**
   * 产品编号
   */
  code: string
  /**
   * 产品名称
   */
  name: string
  /**
   * 需要扣除的信用点数
   */
  credit: number
}

/**
 * 充值套餐
 */
export interface RechargePlan extends Record<string, any> {
  /**
   * 套餐编号
   */
  code: string
  /**
   * 价格
   */
  price: string
  /**
   * 能够获取的信用点数
   */
  credit: number,
  /**
   * 充值套餐名称
   */
  name: string
  /**
   * 购买时发送给客户账单的名称
   */
  buyName:string
  /**
   * 描述信息
   */
  desc: string

  /**
   * 是否推荐
   */
  isMostPop:boolean,

  /**
   * 特性介绍，一行一个
   */
  features:string[]
}
