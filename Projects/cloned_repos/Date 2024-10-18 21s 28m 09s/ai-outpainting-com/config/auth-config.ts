import GoogleProvider from 'next-auth/providers/google'
import { NextAuthConfig } from 'next-auth'

// 将AuthConfig配置与NextAuth分离，目的是在middleware当中使用时不要引用到PrismaAdapter相关文件，导致提示edge不可用PrismaAdapter
// 不要配置与数据库有关的session

let options: any = {
  clientId: process.env.UE_GOOGLE_CLIENT_ID,
  clientSecret: process.env.UE_GOOGLE_CLIENT_SECRET
}
if (process.env.NODE_ENV === 'development') {
  options.issuer = 'http://google.ueboot.com'
  options.userinfo = {
    url: 'http://openidconnect.ueboot.com/v1/userinfo'
  }
  options.token = {
    url: 'http://oauth2-google.ueboot.com/token'
  }
}


export default {
  providers: [
    GoogleProvider(options)
  ],
  cookies: {
    pkceCodeVerifier: {
      name: 'next-auth.pkce.code_verifier',
      options: {
        httpOnly: true,
        sameSite: 'none',
        path: '/',
        secure: true
      }
    }
  }

} satisfies NextAuthConfig
