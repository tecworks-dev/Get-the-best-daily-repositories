'use client';

import Image from 'next/image';
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet';
import { ChevronRight, Minus, Plus } from 'lucide-react';
import { useCart } from './cart-context';
import { SIZES } from './add-to-cart';

export function Cart({ isOpen, onClose }: { isOpen: boolean; onClose: any }) {
  const { items, updateQuantity, total } = useCart();

  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent className="w-full sm:max-w-md border-l-0 p-0 [&_button[aria-label=Close]]:size-12 absolute right-0 top-0 h-full">
        <SheetTitle className="sr-only">Cart</SheetTitle>
        <div className="flex flex-col h-full">
          <div className="flex-1 overflow-auto py-6 px-8">
            {items.map((item) => (
              <div
                key={`${item.id}-${item.size}`}
                className="flex gap-4 py-6 first:pt-0"
              >
                <div className="relative aspect-square h-[120px] bg-[#FFFFFF]">
                  <Image
                    src={item.image}
                    alt={item.name}
                    fill
                    className="object-contain"
                    loading="eager"
                    decoding="sync"
                  />
                </div>
                <div className="flex flex-col justify-between py-1 flex-1">
                  <div className="space-y-1">
                    <p className="font-mono uppercase text-sm">
                      {item.id.split('-').slice(0, -1).join('-')}
                    </p>
                    <p className="font-mono">
                      $
                      {item.id.startsWith('sk')
                        ? item.id.includes('gray')
                          ? '40'
                          : '20'
                        : '20'}
                    </p>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <p className="font-mono text-sm">SIZE</p>
                      <p className="font-mono text-sm">
                        {SIZES[item.size].label}
                      </p>
                    </div>
                    <div className="flex items-center justify-between">
                      <p className="font-mono text-sm">QTY</p>
                      <div className="flex items-center gap-4">
                        <button
                          className="p-1"
                          onClick={() => updateQuantity(item.id, item.size, -1)}
                        >
                          <Minus className="h-3 w-3" />
                        </button>
                        <span className="font-mono">{item.quantity}</span>
                        <button
                          className="p-1"
                          onClick={() => updateQuantity(item.id, item.size, 1)}
                        >
                          <Plus className="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="border-t">
            <div className="p-8 space-y-4">
              <div className="flex justify-between items-center">
                <p className="font-mono uppercase">Total</p>
                <p className="font-mono">${total}</p>
              </div>
              <p className="font-mono text-sm text-muted-foreground">
                TAX AND SHIPPING NOT INCLUDED
              </p>
              <a
                href="https://www.yeezy.com"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full flex items-center justify-between bg-black text-white p-4 font-mono"
              >
                CONTINUE
                <ChevronRight className="h-4 w-4" />
              </a>
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
