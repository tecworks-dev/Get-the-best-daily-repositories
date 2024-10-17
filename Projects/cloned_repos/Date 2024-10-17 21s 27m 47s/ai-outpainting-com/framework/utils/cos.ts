import CosClient from 'cos-js-sdk-v5'
import CosStS from 'qcloud-cos-sts'
import dayjs from 'dayjs'

export type CosCredential = {
  bucket: string
  region: string
  publicPath: string
} & CosStS.CredentialData

let globalConfig = {
  secretId: process.env.UE_COS_SECRET_ID,
  secretKey: process.env.UE_COS_SECRET_KEY,
  bucket: process.env.UE_COS_BUCKET,
  region: process.env.UE_COS_REGION,
  publicPath: process.env.UE_COS_PUBLIC_PATH,
  durationSeconds: process.env.UE_COS_DURATION_SECONDS,
}
let globalClient: CosClient = null as any
let globalCredentials: CosCredential = null as any

async function getCosClient() {
  if (globalClient) {
    return globalClient
  }
  globalClient = new CosClient({
    SecretId: globalConfig.secretId,
    SecretKey: globalConfig.secretKey,
  })
  return globalClient
}

export async function getCosCredential(): Promise<CosCredential> {
  if (globalCredentials && globalCredentials.expiredTime > dayjs().unix()) {
    return Promise.resolve(globalCredentials)
  }
  console.log('开始编译/api/cos', globalConfig.secretId)
  // 配置参数
  const config = {
    secretId: globalConfig.secretId, // 固定密钥
    secretKey: globalConfig.secretKey, // 固定密钥
    proxy: '',
    host: 'sts.tencentcloudapi.com',
    // endpoint: 'sts.internal.tencentcloudapi.com',
    // 密钥有效期
    durationSeconds: globalConfig.durationSeconds ?? 1800,
    bucket: globalConfig.bucket,
    region: globalConfig.region,
    // 这里改成允许的路径前缀，可以根据自己网站的用户登录态判断允许上传的具体路径，例子： a.jpg 或者 a/* 或者 * (使用通配符*存在重大安全风险, 请谨慎评估使用)
    // allowPrefix: ''
    publicPath: globalConfig.publicPath,
  }
  // const appId = config.bucket!.substr(1 + config.bucket!.lastIndexOf('-'));
  const policy = {
    version: '2.0',
    statement: [
      {
        action: [
          // 简单上传
          'name/cos:PutObject',
          'name/cos:PostObject',
          // 分片上传
          // 'name/cos:InitiateMultipartUpload',
          // 'name/cos:ListMultipartUploads',
          // 'name/cos:ListParts',
          // 'name/cos:UploadPart',
          // 'name/cos:CompleteMultipartUpload',
          // 简单上传和分片，需要以上权限，其他权限列表请看 https://cloud.tencent.com/document/product/436/31923
          // 文本审核任务
          // 'name/ci:CreateAuditingTextJob',
          // 开通媒体处理服务
          // 'name/ci:CreateMediaBucket',
          // 更多数据万象授权可参考：https://cloud.tencent.com/document/product/460/41741
        ],
        effect: 'allow',
        principal: { qcs: ['*'] },
        resource: ['*'],
      },
    ],
  }

  return await new Promise((resolve, reject) => {
    const { bucket, region, publicPath } = config
    CosStS.getCredential(
      {
        secretId: config.secretId!,
        secretKey: config.secretKey!,
        proxy: config.proxy,
        durationSeconds: Number(config.durationSeconds),
        // @ts-ignore
        region: config.region,
        policy: policy,
      },
      function (err, credential) {
        if (err) {
          console.log('获取临时密钥出现 异常', err)
          reject(err)
        } else {
          resolve({
            ...credential,
            bucket: bucket!,
            region: region!,
            publicPath: publicPath!,
          })
        }
      },
    )
  })
}

export async function wrapObjectUrl(
  key?: string | null,
  expires = 3600,
): Promise<string> {
  const client = await getCosClient()
  if (client && key) {
    return new Promise((resolve, reject) => {
      client.getObjectUrl(
        {
          Bucket: globalConfig.bucket!,
          Region: globalConfig.region!,
          Key: key,
          Sign: true,
          Expires: expires, //
        },
        (err, data) => {
          if (err) {
            reject(err)
          } else {
            resolve(data.Url)
          }
        },
      )
    })
  }
  return ''
}

export async function wrapDownloadObjectUrl(
  key?: string | null,
  expires = 3600,
): Promise<string> {
  const client = await getCosClient()
  if (client && key) {
    return new Promise((resolve, reject) => {
      client.getObjectUrl(
        {
          Bucket: globalConfig.bucket!,
          Region: globalConfig.region!,
          Key: key,
          Sign: true,
          Expires: expires, //
        },
        (err, data) => {
          if (err) {
            reject(err)
          } else {
            const url = data.Url
            resolve(
              url +
                (url.includes('?') ? '&' : '?') +
                'response-content-disposition=attachment',
            )
          }
        },
      )
    })
  }
  return ''
}
