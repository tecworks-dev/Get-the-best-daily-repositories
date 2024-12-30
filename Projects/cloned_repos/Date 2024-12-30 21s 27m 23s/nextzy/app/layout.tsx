import './globals.css';
import { GeistMono } from 'geist/font/mono';
import { Metadata, Viewport } from 'next';
import { CartProvider } from '@/components/cart-context';

export const metadata: Metadata = {
  title: 'NEXYZY',
  description: 'Inspired by yeezy.com, built with Next.js.',
};

export const viewport: Viewport = {
  themeColor: '#FFFFFF',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${GeistMono.className}`}>
        <CartProvider>
          <div className="flex flex-col min-h-screen h-screen mx-5 overflow-y-scroll">
            {children}
          </div>
        </CartProvider>
      </body>
    </html>
  );
}
