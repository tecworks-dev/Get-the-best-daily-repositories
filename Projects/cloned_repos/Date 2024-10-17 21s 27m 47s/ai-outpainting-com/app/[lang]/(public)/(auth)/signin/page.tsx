'use client'
import { useForm } from 'react-hook-form'
import md5 from 'blueimp-md5'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Image from 'next/image'

interface SignInForm {
  username: string
  password: string
  rememberMe?: boolean
}

const formatError: Record<string, string> = {
  CredentialsSignin: '用户不存在',
  CallbackRouteError: '用户名或密码错误',
  AccessDenied: '用户名或密码错误',
  Unknown: '服务器发生异常，请稍候再试！'
}

export default function Page() {
  const {
    register,
    handleSubmit,
    formState: { errors }
  } = useForm<SignInForm>()
  const [error, setError] = useState<string>()
  const router = useRouter()

  async function onSubmit(data: SignInForm) {
    data.password = md5(data.password)
    // const res = await signIn("credentials", {
    //     ...data,
    //     redirect: false
    // })
    // if (res && res.ok && !res.error) {
    //     router.push("/")
    // } else {
    //     setError(formatError[res?.error ?? "Unknown"])
    // }
    // console.log("res", JSON.stringify(res));
  }

  return (
    <main className="w-full h-screen flex flex-col items-center justify-center px-4">
      <div className="max-w-sm w-full text-gray-600 space-y-5">
        <div className="text-center pb-8">
          <a href="/">
            <Image
              src="/image/ssc-logo.png"
              width={120}
              height={50}
              alt="SSC logo"
              className="mx-auto"
            />
          </a>
          <div className="mt-5">
            <h3 className="text-gray-800 text-2xl font-bold sm:text-3xl">
              使用你的账号登录
            </h3>
          </div>
        </div>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
          <div>
            <label className="font-medium">用户名</label>
            <input
              {...register('username', { required: true })}
              className="w-full mt-2 px-3 py-2 text-gray-500 bg-transparent outline-none border focus:border-indigo-600 shadow-sm rounded-lg"
            />
          </div>
          <div>
            <label className="font-medium">密码</label>
            <input
              {...register('password', { required: true })}
              type="password"
              className="w-full mt-2 px-3 py-2 text-gray-500 bg-transparent outline-none border focus:border-indigo-600 shadow-sm rounded-lg"
            />
          </div>
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-x-3">
              <input
                {...register('rememberMe')}
                type="checkbox"
                id="remember-me-checkbox"
                className="checkbox-item peer hidden"
              />
              <label
                htmlFor="remember-me-checkbox"
                className="relative flex w-5 h-5 bg-white peer-checked:bg-indigo-600 rounded-md border ring-offset-2 ring-indigo-600 duration-150 peer-active:ring cursor-pointer after:absolute after:inset-x-0 after:top-[3px] after:m-auto after:w-1.5 after:h-2.5 after:border-r-2 after:border-b-2 after:border-white after:rotate-45"
              ></label>
              <label htmlFor="remem ber-me-checkbox" className="cursor-pointer">
                记住我
              </label>
            </div>
            {/*<a href="javascript:void(0)"*/}
            {/*   className="text-center text-indigo-600 hover:text-indigo-500">忘记密码?</a>*/}
          </div>
          {error && (
            <div className="text-sm text-red-500 text-center py-3">{error}</div>
          )}
          <button
            className="w-full px-4 py-2 text-white font-medium bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-600 rounded-lg duration-150">
            登录
          </button>
        </form>

        <p className="text-center">
          没有账号?{' '}
          <a
            href="/signup"
            className="font-medium text-indigo-600 hover:text-indigo-500"
          >
            注册
          </a>
        </p>
      </div>
    </main>
  )
}
