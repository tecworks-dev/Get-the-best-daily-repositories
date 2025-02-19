import React, { useState, useEffect, useRef } from 'react';
import data from '@emoji-mart/data';
import Picker from '@emoji-mart/react';
import { EditOutlined } from '@ant-design/icons';
import { useLocale } from 'next-intl';

interface EmojiPickerProps {
  onEmojiSelect: (emoji: string) => void;
  currentEmoji?: string;
}

export const EmojiPicker: React.FC<EmojiPickerProps> = ({ onEmojiSelect, currentEmoji }) => {
  const locale = useLocale();
  const [isOpen, setIsOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(event.target as Node) &&
        !buttonRef.current?.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleEmojiSelect = (emoji: any) => {
    onEmojiSelect(emoji.native);
    setIsOpen(false);
  };

  return (
    <div className="relative inline-block">
      <div
        ref={buttonRef}
        onClick={() => setIsOpen(!isOpen)}
        className='flex items-center justify-center cursor-pointer text-6xl w-24 h-24 bg-slate-200 rounded-full relative'
      >
        {currentEmoji || '😀'}
        <div style={{ backgroundColor: '#3875F6' }} className='w-8 h-8 rounded-full border-2 border-white absolute bottom-0 right-0 flex items-center justify-center'>
          <EditOutlined className='text-sm' style={{ color: 'white' }} />
        </div>
      </div>

      {isOpen && (
        <div
          ref={popoverRef}
          className="absolute z-50 mt-2"
          style={{
            border: '1px solid #EDEDED',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            borderRadius: '10px',
          }}
        >
          <div className="bg-white rounded-lg overflow-hidden">
            <Picker
              data={data}
              onEmojiSelect={handleEmojiSelect}
              theme="light"
              set="native"
              previewPosition='none'
              locale={locale}
            />
          </div>
        </div>
      )}
    </div>
  );
};