import CosClient, { UploadFileParams } from 'cos-js-sdk-v5'

import { useAsync, useSessionStorage } from 'react-use'
import dayjs from 'dayjs'
import { useMemo } from 'react'

export type CosCredentials = {
  bucket: string
  region: string
  publicPath: string
  /** 密钥的起始时间，是 UNIX 时间戳 */
  startTime: number

  /** 密钥的失效时间，是 UNIX 时间戳 */
  expiredTime: number

  /** 临时云 API 凭据 */
  credentials: {
    /** 临时密钥 Id，可用于计算签名 */
    tmpSecretId: string

    /** 临时密钥 Key，可用于计算签名 */
    tmpSecretKey: string

    /** 请求时需要用的 token 字符串，最终请求 COS API 时，需要放在 Header 的 x-cos-security-token 字段 */
    sessionToken: string
  }
}

export function useCosCredentials(immediate = false) {
  const [credentials, setCredentials] = useSessionStorage<CosCredentials>(
    '__ue_cos_credentials',
    null as any,
  )
  return useAsync(async () => {
    // 判断失效时间
    if (credentials && dayjs().unix() < credentials.expiredTime && !immediate) {
      return Promise.resolve(credentials)
    }
    return await fetch(`/api/cos?t=${new Date().getTime()}`, { method: 'GET' }).then(async (res) => {
      const data = (await res.json()) as CosCredentials
      if (!immediate) {
        setCredentials(data)
      }
      return data
    })
  }, [credentials, immediate])
}

let cosClientInstance: CosClient = null as any

export function useCosClient(immediate: boolean = false): CosClient {
  const { value: cosCredentials } = useCosCredentials(immediate)
  cosClientInstance = useMemo(
    () =>
      new CosClient({
        async getAuthorization(options, callback) {
          const { startTime, expiredTime, credentials } = cosCredentials!
          callback({
            TmpSecretId: credentials.tmpSecretId,
            TmpSecretKey: credentials.tmpSecretKey,
            SecurityToken: credentials.sessionToken,
            // 建议返回服务器时间作为签名的开始时间，避免用户浏览器本地时间偏差过大导致签名错误
            StartTime: startTime, // 时间戳，单位秒，如：1580000000
            ExpiredTime: expiredTime, // 时间戳，单位秒，如：1580000000
          })
        },
      }),
    [cosCredentials],
  )
  return cosClientInstance
}

type CosUploadFileObjectOptions = { key: string } & Pick<
  UploadFileParams,
  'onProgress'
>

export function useCosUploader() {
  const client = useCosClient()
  const { value } = useCosCredentials()

  async function uploadFileObject(
    data: File,
    { key, onProgress }: CosUploadFileObjectOptions,
  ) {
    const { bucket, region, publicPath } = value!
    await client.uploadFile({
      Bucket: bucket,
      Region: region,
      Key: key,
      onProgress,
      Body: data,
    } as CosClient.UploadFileParams)
    return {
      key,
      url: `${publicPath}${key}`,
    }
  }

  return {
    uploadFileObject,
  }
}
