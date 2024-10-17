import { fetchGet } from '@/framework/utils'
import { createUploadFileKey } from '@/framework/components/ue-upload/utils'
import { UeUploadProps } from '@/framework/components'
import { useState } from 'react'
import type { UploadProps } from 'antd'
import { t } from '@lingui/macro'

export default function useS34R2(props: UeUploadProps): Partial<UploadProps> {


  function uploadFile(url: string,
                      file: File,
                      onProgress: (event: ProgressEvent<EventTarget>) => void,
                      onSuccess: (ret: any, xhr: any) => void,
                      onError: (err: Error, ret: any) => void
  ) {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', url, true)
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onSuccess(xhr.response, xhr)
      } else {
        onError(new Error(t`Failed to upload file: ${xhr.statusText}`), xhr.response)
      }
    }
    xhr.onerror = () => {
      onError(new Error(t`Network error or CORS issue.`), null)
    }
    xhr.upload.onprogress = onProgress
    xhr.setRequestHeader('Content-Type', file.type)
    // 对文件进行压缩
   /* compressImage(file, (compressedFile:File) => {
      console.log('Compressed file size:', compressedFile.size);
      xhr.send(compressedFile)
    });*/
    xhr.send(file)
    return {
      abort: () => xhr.abort()
    }
  }

  const compressImage = (file:File, callback:Function) => {
    const reader = new FileReader();

    reader.onload = (event:ProgressEvent<FileReader>) => {
      const img:HTMLImageElement = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

        let scaleRatio = 0.5;
        if (img.width > 1024) {
          scaleRatio = 0.3;
        }
        const maxWidth = img.width * scaleRatio; // Dynamic maxWidth based on original width
        const maxHeight = img.height * scaleRatio; // Dynamic maxHeight maintaining aspect ratio

        let width = img.width;
        let height = img.height;

        // Calculate the new width and height
        if (width > height) {
          if (width > maxWidth) {
            height *= maxWidth / width;
            width = maxWidth;
          }
        } else {
          if (height > maxHeight) {
            width *= maxHeight / height;
            height = maxHeight;
          }
        }

        // Set canvas dimensions
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height)

        // Compress the image
        canvas.toBlob((blob) => {
          if(blob){
            const compressedFile = new File([blob], file.name, {
              type: 'image/jpeg',
              lastModified: Date.now()
            });
            callback(compressedFile);
          }else{
            callback(file)
          }
        }, 'image/jpeg', 0.6); // Adjust compression quality here
      };
      if(event.target && event.target.result){
        if (typeof event.target.result === 'string') {
          img.src = event.target.result
        }
      }
    };

    reader.readAsDataURL(file);
  };


  async function customRequest({ file, onProgress, onSuccess, onError }: any) {
    const action = file.extra.action
    return uploadFile(action, file, onProgress, onSuccess, onError)
  }

  async function handleBeforeUpload(file: any, files: any[]) {
    const key = createUploadFileKey(file!.name, props, true)
    const action = await fetchGet<any>(`/api/s34r2?key=${key}`)
    file.extra = { key, action }
    if (props.onBeforeUpload) {
      return props.onBeforeUpload(file, files)
    }
    return file
  }

  const [uploadConfig, setUploadConfig] = useState<UploadProps>({
    action: '',
    method: 'PUT',
    multiple: false,
    maxCount: 1,
    name: 'data',
    customRequest,
    beforeUpload: handleBeforeUpload
  })


  return uploadConfig
}