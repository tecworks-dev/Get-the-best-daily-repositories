import { User } from '@prisma/client'

export type AuthUser = Pick<User, 'id'>

export interface AuthSession extends Record<string, any> {
  isLogged: boolean
  user?: AuthUser
}

export interface Insertable {
  createBy: string
  createTime: Date | null
}

export interface Updateable {
  updateBy: string
  updateTime: Date | null
}

export type Entity = Insertable & Updateable & Record<string, any>

export type EntityMerge<T extends Entity = Entity> = (user: AuthUser) => T

export type FillStrategyOptions<
  I extends boolean = true,
  U extends boolean = true,
> = { insert: I; update: U }
