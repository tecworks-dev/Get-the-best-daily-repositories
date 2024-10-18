import { ARouter, R } from '@/framework/utils'
import orderService from '@/framework/services/OrderService'
import { $Enums, Order } from '@prisma/client'
import { OrderSubmit } from '@/types'
import { t } from '@lingui/macro'
import { PRODUCTS } from '@/lib/consts/products'
import prisma from '@/config/prisma'
import { pick } from 'radash'
import { customAlphabet, urlAlphabet } from 'nanoid'
import { getAuthUser } from '@/auth'

const genOrderNo = customAlphabet(urlAlphabet, 12)


async function createOrder({ path, options }: OrderSubmit) {
  const authUser = await getAuthUser()
  if(!authUser){
    return R.ok({error:"Please Sign In To Continue "})
  }
  console.log(`authUser:${authUser.id}提交订单，剩余${authUser.credit}点`)
  if (authUser.credit <= 0) {
    return R.ok({error:"Your credit points are insufficient, please recharge first."})
  }
  const [{ code, name }] = PRODUCTS
  const order = {
    orderNo: genOrderNo(),
    userId: authUser.id,
    nickname:authUser.name,
    productCode: code,
    productName: name,
    inputImagePath: path,
    outputOptions: JSON.stringify(options ?? {}),
    status: $Enums.OrderStatus.PENDING,
    reason: '',
    createTime: new Date()
  } as Order
  await prisma.$transaction(async tx => {
    await tx.order.create({ data: order })
    // 余额预先扣除
    await tx.user.update({
      data: {
        credit: authUser.credit - 1
      },
      where: {
        id:authUser.id
      }
    })
  })
  console.info(`创建订单：${order.orderNo} --> ${order.status}`)
  const data = pick(order, ['orderNo', 'status', 'createTime'])
  return R.ok(data)
}

const router = new ARouter('/api/order')

router.get('/get', async request => {
  const orderNo = request.nextUrl.searchParams.get('orderNo')
  if (orderNo) {
    const res = (await orderService.getByOrderNo(orderNo)) as Order
    return R.ok(res)
  }
  return R.ok()
})

router.get('/status', async request => {
  const orderNo = request.nextUrl.searchParams.get('orderNo')
  if (orderNo) {
    const res = await orderService.getStatusByOrderNo(orderNo)
    return R.ok(res)
  }
  return R.ok()
})

router.get('/history', async request => {
  const orders = await orderService.getOrderHistory()
  return R.ok(orders)
})

router.post('/submit', async (request) => {
  const body = await request.json()
  return createOrder(body)
})

router.post('/update', async (request) => {
  const body = await request.json()
  return updateOrder(body)
})

async function updateOrder(body: { orderNo: string, costTime: number,outputImagePath:string,status:string,reason:string }) {
  const { orderNo,costTime,outputImagePath,status,reason } = body
  console.log("接收到订单更新消息updateOrder",body)
  const order = await orderService.getByOrderNo(orderNo)
  // 只有处理中的订单才能更新，避免多次发起更新操作，导致多次增加信用点数
  if (order && order.status === $Enums.OrderStatus.PENDING) {
    // 如果状态是失败的，则将信用点数加回
    if(status === $Enums.OrderStatus.FAILED){
      const authUser = await prisma.user.findUnique({where:{id:order.userId}})
      if(authUser){
        console.log(`更新订单updateOrder:${orderNo}，用户：${authUser.id}，剩余${authUser.credit}点`)
        await prisma.user.update({
          data: {
            credit: authUser.credit + 1
          },
          where: {
            id: authUser.id
          }
        })
      }
    }
    await orderService.updateById({costTime,outputImagePath,status:status as $Enums.OrderStatus,reason} as Order,order.id)
  }
  return R.ok()
}

export const { GET, POST, PUT, DELETE } = router.end()