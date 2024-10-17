// 防止编译时变成静态编译，当关联的代码有用到process.env时会变成静态编译
export const dynamic = 'force-dynamic'

export { GET } from '@/framework/apis/cos'
