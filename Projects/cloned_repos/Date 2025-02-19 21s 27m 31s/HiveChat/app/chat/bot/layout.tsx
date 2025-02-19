'use client';
import InPageCollapsed from '@/app/components/InPageCollapsed';

export default function BotsLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <div className='w-full h-22 p-4'>
        <InPageCollapsed />
      </div>
      {children}
    </>
  );
}
