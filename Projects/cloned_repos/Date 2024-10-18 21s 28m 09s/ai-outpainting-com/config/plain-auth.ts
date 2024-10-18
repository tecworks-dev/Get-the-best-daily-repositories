import { getIronSession } from 'iron-session'
import { cookies } from 'next/headers'
import prisma from '@/config/prisma'
import { User } from '@prisma/client'
import { AuthSession, AuthUser, Entity, EntityMerge, FillStrategyOptions } from '@/types'
import { nanoid } from 'nanoid'
import { isFunction } from 'radash'
import services from '@/lib/admin/services/UserService'

const SECRET_KEY =
  'oolfzoXz9Kcv3tCE8ooKCiq4HJavVbRczlwRSftMjpWQwUk9Nd7S22HASdGqOzc9sO3O2'
export const COOKIE_NAME = '_ue_session_id'
/**
 * 会话配置，长久有效
 */
const sessionOptions = {
  password: SECRET_KEY,
  cookieName: COOKIE_NAME,
  ttl: 0
}

/**
 * 解密Cookies中的数据
 * @param cookies
 */
async function decryptCookies(cookies: any): Promise<AuthUser> {
  const { user } = await getIronSession<AuthSession>(cookies, sessionOptions)
  return user!
}

/**
 * 获取会话中的用户信息，仅支持在服务端或服务端组件中使用
 */
export async function getAuthUser(): Promise<User> {
  return getAuthUserInfo()
}

export async function getAuthUserInfo(): Promise<
  User & { cookieValue?: string }
> {
  // const { user } = await getIronSession<AuthSession>(cookies(), sessionOptions)
  const cookie = cookies().get(COOKIE_NAME)
  if (cookie && cookie.value) {
    const entity = await services.getByEmail(cookie.value)
    if (entity) {
      return entity
    }
  }
  const uid = nanoid(128)
  const user = await prisma.user.create({
    data: {
      // nickname: nicknames(),
      // username: nanoid(12),
      // type: 'ANONYMOUS_USER',
      // credit: FREE_GIVE_CREDITS,
      // createTime: new Date(),
      // updateTime: new Date()
    } as User
  })
  return {
    ...user,
    cookieValue: uid
  }
}

export async function fillStrategy<
  T extends Entity = Entity,
  I extends boolean = true,
  U extends boolean = true,
>(data: T | EntityMerge<T>, options?: FillStrategyOptions<I, U>): Promise<T> {
  const { insert, update } = options ?? { insert: true, update: true }
  const user = await getAuthUser()
  let result: T
  if (isFunction(data)) {
    result = data(user)
  } else {
    result = data as T
  }
  if (insert) {
    result['createBy'] = user?.nickname ?? ''
    result['createTime'] = new Date()
  }
  if (update) {
    result['updateBy'] = user?.nickname ?? ''
    result['updateTime'] = new Date()
  }
  return result
}

export async function fillInsertStrategy<T extends Entity = Entity>(
  data: T | EntityMerge<T>
): Promise<T> {
  return fillStrategy<T, true, false>(data)
}

export async function fillUpdateStrategy<T extends Entity = Entity>(
  data: T | EntityMerge<T>
): Promise<T> {
  return fillStrategy<T, false, true>(data)
}
