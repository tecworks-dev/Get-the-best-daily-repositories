"use client";
import React from 'react';
import { usePathname } from 'next/navigation';

const Page = () => {
  const pathname = usePathname();
  return (
    <div></div>
  )
}

export default Page