import { NextRequest, NextResponse } from 'next/server'

export function toQueryString(data: Record<string, any>): string {
  if (data) {
    return new URLSearchParams(data).toString()
  }
  return ''
}

export function toQueryParams(request: NextRequest): Record<string, any> {
  const params = request.nextUrl.searchParams
  return Array.from(params.entries()).reduce(
    (it, [key, val]) => {
      it[key] = val
      return it
    },
    {} as Record<string, any>,
  )
}

export async function fetchJson<T = unknown>(
  input: RequestInfo,
  init?: RequestInit,
): Promise<T> {
  return fetch(input, {
    headers: {
      accept: 'application/json',
      'content-type': 'application/json',
    },
    ...init,
  }).then(async (res) => {
    if (res.ok && 200 === res.status) {
      const data = await res?.json()
      if (200 === data['code']) {
        return data['data']
      } else if (data['message']) {
        return Promise.reject(data['message'])
      }
      return data
    }
    return Promise.reject(res?.statusText ?? 'InternalServerError')
  })
}

export async function fetchGet<T = unknown>(
  url: string,
  params?: Record<string, any>,
): Promise<T> {
  let urlWithSearchParams = url
  if (params) {
    urlWithSearchParams = `${url}?${new URLSearchParams(params).toString()}`
  }
  return fetchJson<T>(urlWithSearchParams, {
    method: 'GET',
  })
}

export async function fetchPost<T = unknown>(
  url: string,
  body: any,
): Promise<T> {
  return fetchJson<T>(url, {
    method: 'POST',
    body: typeof body === 'string' ? body : JSON.stringify(body),
  })
}

export async function fetchPut<T = unknown>(
  url: string,
  params?: Record<string, any>,
): Promise<T> {
  let urlWithSearchParams = url
  if (params) {
    urlWithSearchParams = `${url}?${new URLSearchParams(params).toString()}`
  }
  return fetchJson<T>(urlWithSearchParams, {
    method: 'PUT',
  })
}

export async function fetchDelete<T = unknown>(
  url: string,
  params?: Record<string, any>,
): Promise<T> {
  let urlWithSearchParams = url
  if (params) {
    urlWithSearchParams = `${url}?${new URLSearchParams(params).toString()}`
  }
  return fetchJson<T>(urlWithSearchParams, {
    method: 'DELETE',
  })
}

export function wrapPathRoutes<
  T extends Request = NextRequest,
  K extends string = 'path',
>(routes: [string, (req: T, ctx?: unknown) => any][]) {
  return function (request: T, ctx: { params: { K: string[] } }) {
    let path = ''
    if (ctx.params.K && ctx.params.K.length > 0) {
      path =
        ctx.params.K.length === 1 ? ctx.params.K[0] : ctx.params.K.join('/')
    }
    for (let route of routes) {
      const [rp, handle] = route
      if (rp === path) {
        return handle(request, ctx)
      }
    }
    return R.error(`[${path}] route path not found!`)
  }
}

export const R = {
  raw<T = {}>(body: T) {
    return NextResponse.json(body ?? {})
  },
  ok(data: any = {}) {
    return R.raw({
      code: 200,
      message: 'ok',
      data,
    })
  },
  bad(message: string = '请求参数错误') {
    return R.raw({
      code: 400,
      message,
    })
  },
  error(message: string = '服务器错误') {
    return R.raw({
      code: 500,
      message,
    })
  },
}
