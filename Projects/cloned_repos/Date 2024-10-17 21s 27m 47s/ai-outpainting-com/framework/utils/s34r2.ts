import { GetObjectCommand, PutObjectCommand, S3Client } from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'

let globalClient: S3Client = null as any

function getS3Client() {
  if (globalClient) {
    return globalClient
  }
  globalClient = new S3Client({
    endpoint: process.env.UE_S3_ENDPOINT,
    region: process.env.UE_S3_REGION,
    credentials: {
      accessKeyId: process.env.UE_S3_ACCESS_KEY!,
      secretAccessKey: process.env.UE_S3_SECRET_KEY!
    }
  })
  return globalClient
}

export function getBucket() {
  return process.env.UE_S3_BUCKET ?? 'ai-outpainting'
}

export function createPutSingedUrl(key: string, customBucket?: string, expiresIn = 3600) {
  const bucket = customBucket ? customBucket : getBucket()
  const command = new PutObjectCommand({ Bucket: bucket, Key: key! })
  const client = getS3Client()
  return  getSignedUrl(client, command, {
    expiresIn
  })
}

export async function createGetSingedUrl(key?: string | null, customBucket?: string, expiresIn = 3600) {
  if (!key) {
    return key
  }
  const bucket = customBucket ? customBucket : getBucket()
  const command = new GetObjectCommand({ Bucket: bucket, Key: key })
  const client = getS3Client()
  return await getSignedUrl(client, command, {
    expiresIn
  })
}
