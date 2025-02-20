"use client";
import React from 'react';
import { MessageList } from '@/app/components/MessageList';
export default function Chat({ params }: { params: { chat_id: string } }) {
  const chatId = params.chat_id;
  return (
    <MessageList chat_id={chatId} />
  );
}
