import { $Enums, Order } from '@prisma/client'
import prisma from '@/config/prisma'
import { isNumber } from 'radash'
import { getAuthUser } from '@/auth'
import { createGetSingedUrl } from '@/framework/utils/s34r2'

const services = {
  async list(data: Partial<Order>) {
    return prisma.order.findMany({
      where: { ...data }
    })
  },
  async getById(id: number) {
    return prisma.order.findUnique({
      where: {
        id
      }
    })
  },
  async getStatusByOrderNo(orderNo: string) {
    return prisma.order
      .findUnique({
        select: {
          orderNo: true,
          status: true,
          createTime: true
        },
        where: {
          orderNo
        }
      })
  },
  async getByOrderNo(orderNo: string) {
    const order = await prisma.order
      .findUnique({
        where: {
          orderNo
        }
      })
    return order ?
      {
        ...order,
      }
      : order
  },
  async getOrderHistory() {
    const authUser = await getAuthUser()
    // 如果未登录，则返回空
    if(!authUser){
      return []
    }
    const res = await prisma.order.findMany({
      skip: 0,
      take: 30,
      select: {
        orderNo: true,
        inputImagePath: true,
        outputImagePath: true
      },
      where: {
        userId: authUser.id,
        status: $Enums.OrderStatus.SUCCEED
      },
      orderBy: [
        {
          id: 'desc'
        }
      ]
    })
    return res.map(it => ({
      ...it,
      inputImageUrl: createGetSingedUrl(it.inputImagePath),
      outputImageUrl: createGetSingedUrl(it.outputImagePath)
    }))
  },
  async updateById(data: Order,id:number) {
    return prisma.order.update({
      data,
      where: {
        id
      }
    })
  },
  async deleteById(data: Order | number) {
    const id = isNumber(data) ? data : data.id
    return prisma.order.delete({
      where: {
        id
      }
    })
  }
}

export default services
