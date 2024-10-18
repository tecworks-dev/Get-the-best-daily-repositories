import { offsetPage, stripPage } from '../utils'
import { PageQueryable } from '../types'
import { DictData } from '@prisma/client'
import prisma from '@/config/prisma'
import { fillInsertStrategy, fillUpdateStrategy } from '@/config/plain-auth'
import { unstable_noStore as noStore } from 'next/cache'

const service = {
  async paging(query: PageQueryable<DictData>) {
    const { page, params } = stripPage<PageQueryable<DictData>>(query)
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
      prisma.dictData.count({ where }),
      prisma.dictData.findMany({
        ...offsetPage(page),
        where,
        orderBy: {
          sort: 'desc',
        },
      }),
    ])
  },
  async create(data: DictData) {
    return prisma.dictData.create({
      data: await fillInsertStrategy(data),
    })
  },
  async findById(id: number) {
    noStore()
    return prisma.dictData.findFirst({ where: { id } })
  },
  async update(data: DictData) {
    return prisma.dictData.update({
      data: await fillUpdateStrategy(data),
      where: {
        id: data.id,
      },
    })
  },
  async getByValue(type: string, value: string) {
    noStore()
    return prisma.dictData.findFirst({
      where: {
        type,
        value,
      },
    })
  },
  async delete(id: number) {
    return prisma.dictData.delete({ where: { id } })
  },
  async findByDictType(dictType: string) {
    return prisma.dictData.findMany({
      where: {
        type: {
          equals: dictType,
        },
      },
      orderBy: {
        sort: 'desc',
      },
    })
  },
}
export default service
