import { useState, useCallback } from 'react';
import { message } from "antd";
import { useTranslations } from 'next-intl';

const useImageUpload = (maxImages: number = 5) => {
  const t = useTranslations('Chat');
  const [uploadedImages, setUploadedImages] = useState<Array<{ url: string; file: File }>>([]);

  const handleImageUpload = useCallback(async (file?: File, url?: string) => {
    if (file && url) {
      // 直接处理传入的文件
      if (file.size > 5 * 1024 * 1024) { // 5MB
        message.warning(t('imageSizeLimit'));
        return;
      }
      if (!file.type.startsWith('image/')) {
        message.warning(t('mustBeImage'));
        return;
      }

      setUploadedImages(prev => [...prev, { url, file }]);
      return;
    }

    // 原有的文件选择逻辑
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.multiple = true;

    input.onchange = async (e) => {
      const files = (e.target as HTMLInputElement).files;
      if (!files) return;

      const fileArray = Array.from(files);
      if (fileArray.length + uploadedImages.length > maxImages) {
        message.warning(t('maxImageCount', { maxImages: maxImages }));
        return;
      }

      // 验证文件大小和类型
      for (const file of fileArray) {
        if (file.size > 5 * 1024 * 1024) { // 5MB
          message.warning(t('imageSizeLimit'));
          return;
        }
        if (!file.type.startsWith('image/')) {
          message.warning(t('mustBeImage'));
          return;
        }
      }
      const newImages = fileArray.map(file => ({
        url: URL.createObjectURL(file),
        file
      }));
      setUploadedImages(prev => [...prev, ...newImages]);
    };
    input.click();
  }, [uploadedImages.length, maxImages, t]);

  const removeImage = useCallback((index: number) => {
    setUploadedImages(prev => {
      const imgToRemove = prev[index];
      if (imgToRemove.url.startsWith('blob:')) {
        URL.revokeObjectURL(imgToRemove.url);
      }
      return prev.filter((_, i) => i !== index);
      // return prev.filter((item) => item.url !== imgToRemove.url);
    });
  }, []);

  return { uploadedImages, maxImages, handleImageUpload, removeImage, setUploadedImages };
};

export default useImageUpload;