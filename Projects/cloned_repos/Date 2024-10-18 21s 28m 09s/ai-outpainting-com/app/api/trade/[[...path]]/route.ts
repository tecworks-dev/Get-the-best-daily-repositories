import { PUT, wrapGetRouter, wrapPostRouter } from '@/framework/apis/TradeApi'
import { RECHARGES } from '@/lib/consts/products'

const GET = wrapGetRouter(RECHARGES)
const POST = wrapPostRouter(RECHARGES)

export { GET, POST, PUT }
