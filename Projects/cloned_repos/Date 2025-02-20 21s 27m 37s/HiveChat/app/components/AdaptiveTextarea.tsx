"use client";
import React, { useState, useRef, useEffect } from 'react';
import { Button, Tooltip, message, Image as AntdImage } from 'antd';
import useImageUpload from '@/app/hooks/chat/useImageUpload';
import ImageIcon from "@/app/images/image.svg";
import CloseIcon from '@/app/images/close.svg'
import { fileToBase64 } from '@/app/utils';
import { ArrowUpOutlined } from '@ant-design/icons';
import { LLMModel } from '@/app/adapter/interface';
import clsx from 'clsx';
import { useTranslations } from 'next-intl';

const AdaptiveTextarea = (props: {
  model: LLMModel
  submit: (message: string, attachments?: Array<{ mimeType: string; data: string }>) => void
}) => {
  const t = useTranslations('Chat');
  const [text, setText] = useState('');
  const [submitBtnDisable, setSubmitBtnDisable] = useState(true);
  const [pending, setPending] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const testSpanRef = useRef<HTMLSpanElement | null>(null);
  const [isOverflow, setIsOverflow] = useState(false);

  const { uploadedImages, maxImages, handleImageUpload, removeImage } = useImageUpload();
  // 创建测试 span
  useEffect(() => {
    // 创建用于测量文本宽度的 span 元素
    const span = document.createElement('span');
    // span.style.visibility = 'hidden';
    // span.style.position = 'absolute';
    const holder = document.getElementById('test-span-holder');
    if (holder) {
      holder.appendChild(span);
    }
    testSpanRef.current = span;
    // 组件卸载时移除 span
    return () => {
      if (testSpanRef.current) {
        const holder = document.getElementById('test-span-holder');
        holder?.removeChild(testSpanRef.current);
      }
    };
  }, []);

  // 检查文本是否溢出
  const checkOverflow = () => {
    if (textareaRef.current && containerRef.current && testSpanRef.current) {
      const textarea = textareaRef.current;
      const container = containerRef.current;
      const testSpan = testSpanRef.current;

      // 更新测试 span 的样式和内容
      testSpan.style.font = window.getComputedStyle(textarea).font;
      testSpan.textContent = textarea.value;

      const testHeightDiv = document.getElementById('test-span-holder')
      const testHeight = testHeightDiv?.scrollHeight || 16;
      // 处理 rows
      const padding = 12;
      const lineHeight = 24;
      const minRows = 1;     // 最小行数
      const maxRows = 4;     // 最大行数
      // const contentHeight = testHeight - padding * 2;
      const contentHeight = testHeight;
      const contentRows = Math.ceil(contentHeight / lineHeight);

      // 计算换行符的数量
      const lineBreaks = (textarea.value.match(/\n/g) || []).length;
      const totalRows = contentRows + lineBreaks; // 总行数包括换行符
      // 限制行数在 minRows 和 maxRows 之间
      const rows = Math.min(Math.max(totalRows, minRows), maxRows);
      textarea.rows = rows;

      // 计算文本实际宽度是否超过容器可用宽度
      const textWidth = testSpan.offsetWidth;
      const availableWidth = container.offsetWidth - 120; // 减去按钮宽度和间距
      if (totalRows > 1) {
        setIsOverflow(true);
      } else {
        setIsOverflow(textWidth > availableWidth);
      }
    }
  };

  // 监听文本变化
  useEffect(() => {
    checkOverflow();
    if (text.trim() === '') {
      setSubmitBtnDisable(true);
    } else {
      setSubmitBtnDisable(false);
    }
  }, [text]);

  // 监听窗口大小变化
  useEffect(() => {
    const handleResize = () => {
      checkOverflow();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto p-4">
      <div
        ref={containerRef}
        className={clsx({ 'bg-gray-100': pending }, 'flex border-gray-300  border rounded-3xl p-2 flex-col justify-end')}
      >
        {uploadedImages.length > 0 && <div className="h-20 flex flex-col justify-center items-center">
          <div className='flex flex-row h-16 max-w-3xl pl-2 w-full'>
            {uploadedImages.map((image, index) => (
              <div key={index} className="relative group mr-4 h-16 w-16">
                <AntdImage alt=''
                  className='block border h-full w-full rounded-md object-cover cursor-pointer'
                  height={64}
                  width={64}
                  src={image.url}
                  preview={{
                    mask: false
                  }}
                />
                <div
                  className="absolute bg-white rounded-full -top-2 -right-2 cursor-pointer hidden group-hover:block"
                  onClick={() => removeImage(index)}
                >
                  <CloseIcon className='w-5 h-5' alt='close' />
                </div>
              </div>
            ))}
          </div>
        </div>}
        <div className={`flex ${isOverflow ? 'flex-col justify-end' : 'flex-row items-center'}`}>
          <textarea
            ref={textareaRef}
            autoFocus={true}
            value={text}
            rows={1}
            onChange={(e) => setText(e.target.value)}
            placeholder={t('inputPlaceholder')}
            className={clsx({ 'bg-gray-100': pending }, "flex-1 p-2  leading-6 h-10 py-2 px-3 rounded-md outline-none resize-none")}
            disabled={pending}
            style={{
              scrollbarWidth: 'thin',
              scrollbarColor: '#eaeaea transparent'
            }}
            onPaste={async (e) => {
              // 阻止默认粘贴行为
              if (e.clipboardData.files.length > 0) {
                e.preventDefault();
                if (!props.model.supportVision) {
                  message.warning(t('notSupportVision'));
                  return;
                }

                const files = Array.from(e.clipboardData.files);
                // 检查是否为图片文件
                const imageFiles = files.filter(file => file.type.startsWith('image/'));

                if (imageFiles.length > 0) {
                  if (uploadedImages.length + imageFiles.length > maxImages) {
                    message.warning(t('maxImageCount', { maxImages: maxImages }));
                    return;
                  }
                  // 调用已有的图片上传处理函数
                  for (const file of imageFiles) {
                    const url = URL.createObjectURL(file);
                    // 使用已有的 uploadedImages 状态更新函数
                    handleImageUpload(file, url);
                  }
                }
              }
            }}
            onKeyDown={async (e) => {
              // 如果是在输入法编辑状态，直接返回，不做处理
              if (e.nativeEvent.isComposing) {
                return;
              }
              if (e.key === 'Enter') {
                if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) {
                  // Command/Ctrl + Enter: 插入换行
                  e.preventDefault();
                  const target = e.currentTarget;
                  const start = target.selectionStart;
                  const end = target.selectionEnd;
                  const newValue = target.value.substring(0, start) + '\n' + target.value.substring(end);

                  // 创建一个合成事件来更新输入
                  setText(newValue);

                  // 下一个事件循环设置光标位置
                  setTimeout(() => {
                    target.selectionStart = target.selectionEnd = start + 1;
                  }, 0);
                  return;
                }
                e.preventDefault();
                if (text.trim() === '') {
                  return;
                }
                setPending(true);
                props.submit(text, await Promise.all(uploadedImages
                  .filter(img => img.file)
                  .map(async (img) => {
                    return {
                      mimeType: img.file.type,
                      data: await fileToBase64(img.file!)
                    }
                  })))
              }
            }}
          />
          <div className={clsx({ 'w-full justify-end': isOverflow }, 'flex flex-row items-center')}>
            {
              props.model?.supportVision ?
                <Button
                  onClick={() => handleImageUpload()}
                  type='text'
                  className='mr-2'
                  icon={<ImageIcon style={{ verticalAlign: 'middle' }} width={32} height={32} />}
                />
                :
                <Tooltip title={t('notSupportVision')}>
                  <Button
                    type='text'
                    disabled={true}
                    className='mr-2'
                    icon={<ImageIcon style={{ verticalAlign: 'middle' }} width={32} height={32} />}
                  />
                </Tooltip>
            }

            {(pending || submitBtnDisable) ?
              <Button
                type="primary"
                style={{ backgroundColor: '#6ba7fc', color: '#fff', border: 'none' }}
                disabled
                shape="circle"
                icon={<ArrowUpOutlined color='#fff' />} />
              :
              <Button
                type="primary"
                shape="circle"
                onClick={async () => {
                  setPending(true);
                  props.submit(text, await Promise.all(uploadedImages
                    .filter(img => img.file)
                    .map(async (img) => {
                      return {
                        mimeType: img.file.type,
                        data: await fileToBase64(img.file!)
                      }
                    })))
                }}
                icon={<ArrowUpOutlined />} />}
          </div>
        </div>
      </div>
      <div className='mt-10 bg-gray-200 px-5 leading-6 border h-6 invisible' id='test-span-holder'></div>
    </div>
  );
};

export default AdaptiveTextarea;