import NextAuth from 'next-auth'
import prisma from '@/config/prisma'
import { PrismaAdapter } from '@auth/prisma-adapter'
import authConfig from '@/config/auth-config'
import { SessionUser } from '@/framework/types/sessionUser'
import userService from '@/lib/admin/services/UserService'
import { User, UserType } from '@prisma/client'
import JWTUtils  from "@/framework/utils/JWTUtils"
import type { ResponseCookie } from "next/dist/compiled/@edge-runtime/cookies"
import { type NextRequest, NextResponse } from "next/server"
import { getCookie, setCookie } from "cookies-next"
import { cookies } from "next/headers"

const Give_Credit = 5

export const { handlers, signIn, signOut, auth } = NextAuth({
  debug:process.env.NODE_ENV !== "production",
  adapter: PrismaAdapter(prisma),
  session: {
    strategy: 'jwt'
  },
  callbacks: {
    async signIn({ account, profile }) {
      console.log(`##用户:${profile?.email}登录系统`,profile)
      if (account?.provider === 'google' && profile?.email) {
        return true
      }
      return `This authorization method is not supported`
    },
    async session({ session, user }) {
      const dbUser = (await userService.getByEmail(session.user.email))!
      // 第一次登录赠送点数
      if (dbUser && !dbUser.giveCredit) {
        dbUser.credit = Give_Credit
        dbUser.totalCredit = Give_Credit
        dbUser.giveCredit = true
        console.log(`##用户:${dbUser?.email} 首次登录系统，赠送 ${dbUser.giveCredit} Credit `)
        await  userService.updateByEmail(dbUser,dbUser.email)
      }
      session.user = {
        id: dbUser.id,
        name: dbUser.name,
        email: dbUser.email,
        image:dbUser.image,
        emailVerified: new Date(),
        credit: dbUser.credit ?? 0
      } as SessionUser
      console.log("###获取当前登录用户信息User",session.user)
      return session
    }
  },
  ...authConfig
})
// 获取当前已绑定google的登录用户信息
export const getAuthUser = async () => {
  const session = await auth()
  console.log("session",session)
  return session?.user! as SessionUser
}

// ============================ 临时用户创建相关代码 ============================

export interface AuthUser extends Omit<User, "id"> {
	userId: string
	tempUserId?: string
	isTempUser: boolean
}

export const TOKEN_COOKIE_NAME = "ue_x_token"
export const USER_COOKIE_NAME = "ue_x_user"

export const cookieOptions: Partial<ResponseCookie> = {
	httpOnly: true,
	sameSite: "lax",
	path: "/",
	// 2099年12月31日
	expires: new Date(4089668577000),
	secure: process.env.NODE_ENV === "production",
}


export async function createUser(request: NextRequest) {
	const response = NextResponse.next()
	if (!request.cookies.has(USER_COOKIE_NAME)) {
		try {
			// 创建临时用户信息,默认userId=0,后续真实操作业务时再入库
			const tempUser = await JWTUtils.genTempUser()
			const token = await JWTUtils.encode(tempUser)
			response.cookies.set(TOKEN_COOKIE_NAME, token, cookieOptions)
			response.cookies.set(
				USER_COOKIE_NAME,
				JSON.stringify(tempUser),
				cookieOptions,
			)
		} catch (e) {
			console.error("create user failed: ", e)
		}
	}
	return response // 确保返回 response
}


// 获取当前用户信息，如果是临时用户则插入真实User，但不绑定google，并修改cookie当中的token
export async function getCurrentUser(
	throws = false,
): Promise<AuthUser | undefined> {
	const auth = await authUser()
	if (!auth) {
		// 如果 auth 为 undefined，直接创建新用户
		return await createRealUser()
	}
	
	if (!auth.isTempUser) {
    // 查询数据库获取用户信息
    const user = await userService.getById(auth.userId)
    if(user){
      // console.log("###获取当前登录用户信息User",user)
      return {userId:user.id,isTempUser:false,...user}
    }
	}

	// 创建真实用户并更新 auth
	return await createRealUser(auth)
}

export async function authUser() {
	const auth = getCookie(USER_COOKIE_NAME, { cookies })
	try {
		if (auth) {
			// 读取旧的 Cookie 信息
			return JSON.parse(auth!) as AuthUser
		} else {
			console.log("读取 authUser 失败,cookie为空")
			return undefined
		}
	} catch (e) {
		console.log("读取 authUser 失败", e)
		return undefined
	}
}


async function createRealUser(existingAuth?: AuthUser): Promise<AuthUser> {
	try {
   
		const user = await prisma.user.create({
			data: {
        type: UserType.ANONYMOUS_USER,
        name: "",
        email: "",
        image: "",
        emailVerified: new Date(),
        credit: Give_Credit,
        totalCredit: Give_Credit,
        giveCredit: true,
      },
		})
    console.log(`##用户:${user?.id} 首次登录系统，赠送 ${user.giveCredit} Credit `)

		const updatedAuth = {
			...(existingAuth || {createTime: new Date(),email: ""}),
      ...user,
			userId: user.id,
			isTempUser: false,
		} as AuthUser

		const token = await JWTUtils.encode(updatedAuth)
		setCookie(TOKEN_COOKIE_NAME, token, { cookies, ...cookieOptions })
		setCookie(USER_COOKIE_NAME, JSON.stringify(updatedAuth), {
			cookies,
			...cookieOptions,
		})
		return updatedAuth
	} catch (error) {
		console.error("创建真实用户失败", error)
		throw error
	}
}