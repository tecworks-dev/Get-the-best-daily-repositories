import { $Enums, Trade } from '@prisma/client'
import prisma from '@/config/prisma'
import { isNumber } from 'radash'
import { RechargePlan } from '@/framework/types'
import { getAuthUser } from '@/auth'
import { nanoid } from 'nanoid'

const services = {
  async list(data: Trade) {
    return prisma.trade.findMany({
      where: { ...data },
    })
  },
  async getByTradeNo(tradeNo: string) {
    return prisma.trade.findUnique({
      where: {
        tradeNo,
      },
    })
  },
  async getByOutTradeNo(outTradeNo:string){
    return prisma.trade.findFirst({
      where: {
        outTradeNo,
      },
    })
  },
  async createTrade(data: RechargePlan, channel: string) {
    const user = await getAuthUser()
    return prisma.trade.create({
      data: {
        userId: user.id,
        tradeNo: nanoid(32),
        userEmail:user.email,
        planCode: data.code,
        planName: data.name,
        channel,
        credit: data.credit,
        details: '',
        amount: data.price,
        status: $Enums.TradeStatus.WAIT_PAID,
        outTradeNo: '',
        outTradeStatus: '',
      } as Trade,
    })
  },
  async updateById(data: Trade) {
    const { id } = data
    //
    return prisma.trade.update({
      data,
      where: {
        id,
      },
    })
  },
  // 支付成功回调处理，需要更新订单状态和用户余额
  async paySuccessCallback(data:Trade){
    const { id } = data
    // 相同事物内更新订单状态和用户余额
    return prisma.$transaction(async tx=>{
      await tx.trade.update({
        data,
        where: {
          id,
        },
      })
      const dbUser = await tx.user.findFirst({
        where:{
          id:data.userId
        }
      })
      if(!dbUser){
        console.error(`更新用户余额时，未找到用户ID:${data.userId}`)
        throw new Error(`When update user credit,Can't found user Id:${data.userId}`)
      }
      console.info(`用户Id:${data.userId},email:${data.userEmail},充值套餐前（before），信用点:${dbUser.credit}`)
      dbUser.totalCredit = dbUser.totalCredit+data.credit
      dbUser.credit = dbUser.credit+data.credit
      await tx.user.update({data:dbUser,where:{id:dbUser.id}})
      console.info(`用户Id:${data.userId},email:${data.userEmail},充值套餐后，信用点:${dbUser.credit}`)
    })
  },
  async deleteById(data: Trade | number) {
    const id = isNumber(data) ? data : data.id
    return prisma.trade.delete({
      where: {
        id,
      },
    })
  },
}

export default services
