import prisma from '@/config/prisma'
import { $Enums, User } from '@prisma/client'
import { PageQueryable } from '@/types'
import { offsetPage, stripPage } from '@/utils'
import nicknames from '@/framework/utils/nicknames'
import { nanoid } from 'nanoid'
import { FREE_GIVE_CREDITS } from '@/lib/consts/products'


const services = {
  async paging(query: PageQueryable<User>) {
    const { page, params } = stripPage(query)
    return prisma.$transaction([
      prisma.user.count({
        where: {
          ...params
        }
      }),
      prisma.user.findMany({
        ...offsetPage(page),
        select: {
          id: true,
          type: true,
          nickname: true,
          email: true,
          createdAt: true,
          updatedAt: true
        },
        where: {
          ...params
        }
      })
    ])
  },
  async getByEmail(email: string) {
    return prisma.user.findUnique({
      where: {
        email
      }
    })
  },
  async updateByEmail(user:User,email: string) {
    return prisma.user.update({
      data:user,
      where: {
        email
      }
    })
  },
  // 根据id查询用户
  async getById(id: string) {
    return prisma.user.findUnique({
      where: {
        id
      }
    })
  }
}

export default services
