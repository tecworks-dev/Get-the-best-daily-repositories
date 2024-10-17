import { offsetPage, stripPage } from '@/utils'
import { PageQueryable } from '@/types'
import { Dict } from '@prisma/client'
import prisma from '@/config/prisma'
import { fillInsertStrategy, fillUpdateStrategy } from '@/config/plain-auth'
import { unstable_noStore as noStore } from 'next/dist/server/web/spec-extension/unstable-no-store'

const service = {
  async paging(query: PageQueryable<Dict>) {
    noStore()
    const { page, params } = stripPage<PageQueryable<Dict>>(query)
    // 拼装查询条件
    const where = {
      title: params.title
        ? {
            contains: params.title,
          }
        : undefined,
      type: params.type
        ? {
            contains: params.type,
          }
        : undefined,
    }
    return prisma.$transaction([
      prisma.dict.count({ where }),
      prisma.dict.findMany({
        ...offsetPage(page),
        where,
        orderBy: {
          id: 'desc',
        },
      }),
    ])
  },
  async create(data: Dict) {
    return prisma.dict.create({
      data: await fillInsertStrategy(data),
    })
  },
  async findById(id: number) {
    noStore()
    return prisma.dict.findFirst({ where: { id } })
  },
  async update(data: Dict) {
    return prisma.dict.update({
      data: await fillUpdateStrategy(data),
      where: {
        id: data.id,
      },
    })
  },
  async getByType(type: string) {
    noStore()
    return prisma.dict.findFirst({ where: { type } })
  },

  async delete(id: number) {
    return prisma.dict.delete({ where: { id } })
  },
}
export default service
