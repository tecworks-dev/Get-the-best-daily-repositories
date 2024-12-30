'use client';

import React, { createContext, use, useState, useCallback } from 'react';
import { type Product } from '@/lib/products';

interface CartItem extends Product {
  quantity: number;
  size: number;
}

interface CartContextType {
  items: CartItem[];
  addToCart: (product: Product, size: number) => void;
  updateQuantity: (id: string, size: number, change: number) => void;
  total: number;
}

const CartContext = createContext<CartContextType | undefined>(undefined);

// Want a full-featured optimistic cart?
// Check out Next.js Commerce: https://github.com/vercel/commerce
// Want to use Stripe instead?
// Check out my starter: https://github.com/leerob/next-saas-starter
export function CartProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<CartItem[]>([]);

  const addToCart = useCallback((product: Product, size: number) => {
    setItems((prevItems) => {
      const existingItemIndex = prevItems.findIndex(
        (item) => item.id === product.id && item.size === size,
      );
      if (existingItemIndex > -1) {
        return prevItems.map((item, index) =>
          index === existingItemIndex
            ? { ...item, quantity: item.quantity + 1 }
            : item,
        );
      } else {
        return [...prevItems, { ...product, quantity: 1, size }];
      }
    });
  }, []);

  const updateQuantity = useCallback(
    (id: string, size: number, change: number) => {
      setItems((prevItems) =>
        prevItems.reduce((acc, item) => {
          if (item.id === id && item.size === size) {
            const newQuantity = item.quantity + change;
            return newQuantity > 0
              ? [...acc, { ...item, quantity: newQuantity }]
              : acc;
          }
          return [...acc, item];
        }, [] as CartItem[]),
      );
    },
    [],
  );

  const total = items.reduce((acc, item) => {
    const price = item.id.startsWith('sk')
      ? item.id.includes('gray')
        ? 40
        : 20
      : 20;
    return acc + price * item.quantity;
  }, 0);

  return (
    <CartContext.Provider value={{ items, addToCart, updateQuantity, total }}>
      {children}
    </CartContext.Provider>
  );
}

export function useCart() {
  const context = use(CartContext);
  if (context === undefined) {
    throw new Error('useCart must be used within a CartProvider');
  }
  return context;
}
