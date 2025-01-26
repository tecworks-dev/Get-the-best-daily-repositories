import { R2 } from '@/lib/s3-client';
import { HeadObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3';
import axios from 'axios';
import crypto from 'crypto';

export async function generateFileName(base: string, usePrefix: 'upload' | 'logo' = 'upload') {
  const hash = crypto.createHash('md5')
    .update(base + (usePrefix === 'upload' ? Date.now() : ''))
    .digest('hex')
    .slice(0, 8);
  const fileExtension = base.split('.').pop() || 'jpg';
  return `${usePrefix}-${usePrefix === 'logo' ? base + '-' : ''}${hash}.${fileExtension}`;
}

export async function checkFileExists(filename: string) {
  try {
    await R2.send(new HeadObjectCommand({
      Bucket: process.env.R2_BUCKET_NAME,
      Key: filename,
    }));
    return true;
  } catch {
    return false;
  }
}

export async function uploadToR2(filename: string, data: Buffer | ArrayBuffer) {
  // Convert ArrayBuffer to Buffer if needed
  const bodyData = Buffer.isBuffer(data) ? data : Buffer.from(data);
  await R2.send(new PutObjectCommand({
    Bucket: process.env.R2_BUCKET_NAME,
    Key: filename,
    Body: bodyData,
    ContentType: 'image/jpeg',
    CacheControl: 'public, max-age=31536000',
  }));
  return `${process.env.R2_PUBLIC_URL}/${filename}`;
}

export async function downloadImage(imageUrl: string) {
  const response = await axios.get(imageUrl, {
    responseType: 'arraybuffer'
  });
  return response.data;
}