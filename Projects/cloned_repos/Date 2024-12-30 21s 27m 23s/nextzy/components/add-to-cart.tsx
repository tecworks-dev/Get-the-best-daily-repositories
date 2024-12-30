'use client';

import { useState, useCallback } from 'react';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Cart } from '@/components/cart';
import { type Product } from '@/lib/products';
import { useCart } from './cart-context';

export const SIZES = [
  { label: 'S-M', value: 0 },
  { label: 'M-L', value: 1 },
  { label: 'XL-XXL', value: 2 },
];

export function AddToCart({ product }: { product: Product }) {
  const [isSelectingSize, setIsSelectingSize] = useState(false);
  const [isCartOpen, setIsCartOpen] = useState(false);
  const { addToCart } = useCart();

  const handleAddToCart = useCallback(
    (size: number) => {
      addToCart(product, size);
      setIsSelectingSize(false);
      setIsCartOpen(true);
    },
    [addToCart, product],
  );

  const productName = product.id
    .split('-')
    .slice(0, -1)
    .join('-')
    .toUpperCase();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.3 }}
    >
      <div className="relative w-full max-w-[320px] mx-auto">
        <motion.div
          className="flex flex-col items-center"
          initial={false}
          animate={isSelectingSize ? 'selecting' : 'idle'}
        >
          {/* Product Name / Select Size */}
          <motion.div
            className="h-8 relative w-full flex justify-center items-center overflow-hidden"
            variants={{
              idle: { y: 0 },
              selecting: { y: 0 },
            }}
          >
            <motion.p
              className="font-medium font-mono uppercase absolute inset-0 flex items-center justify-center"
              variants={{
                idle: { y: 0 },
                selecting: { y: '-100%' },
              }}
              transition={{ type: 'tween', ease: 'easeInOut', duration: 0.2 }}
            >
              {productName}
            </motion.p>
            <motion.div
              className="flex items-center justify-between w-full absolute inset-0"
              variants={{
                idle: { y: '100%' },
                selecting: { y: 0 },
              }}
              transition={{ type: 'tween', ease: 'easeInOut', duration: 0.2 }}
            >
              <div className="w-8" />
              <p className="font-medium font-mono uppercase">SELECT SIZE</p>
              <motion.div
                className="size-8"
                variants={{
                  idle: { x: '-50%', y: 28, opacity: 0 },
                  selecting: { x: 0, y: 0, opacity: 1 },
                }}
                transition={{ type: 'spring', stiffness: 300, damping: 25 }}
              >
                <button
                  onClick={() => setIsSelectingSize(false)}
                  className="w-full h-full flex items-center justify-center"
                  aria-label="Close size selector"
                >
                  <X className="size-5" />
                </button>
              </motion.div>
            </motion.div>
          </motion.div>

          {/* Plus Button / Size Buttons */}
          <motion.div className="mt-8 relative w-full h-12">
            <motion.button
              onClick={() => setIsSelectingSize(true)}
              className="size-12 flex items-center justify-center bg-white absolute left-1/2 -translate-x-1/2"
              variants={{
                idle: { opacity: 1 },
                selecting: { opacity: 0 },
              }}
              aria-label="Select size"
              aria-expanded={isSelectingSize}
            >
              <svg
                className="size-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
            </motion.button>

            <AnimatePresence>
              {isSelectingSize && (
                <motion.div
                  className="flex justify-between w-full absolute top-0 left-0"
                  initial="hidden"
                  animate="visible"
                  exit="hidden"
                >
                  {SIZES.map((size, index) => (
                    <motion.button
                      key={size.value}
                      onClick={() => handleAddToCart(size.value)}
                      className="w-16 h-12 bg-transparent hover:bg-black hover:text-white transition-colors font-mono text-sm font-semibold"
                      variants={{
                        hidden: {
                          x: 'calc(50% - 32px)',
                          y: 0,
                          opacity: 0,
                        },
                        visible: {
                          x:
                            index === 0
                              ? '0'
                              : index === 2
                                ? 'calc(100% - 64px)'
                                : 'calc(50% - 32px)',
                          y: 0,
                          opacity: 1,
                          transition: {
                            type: 'spring',
                            stiffness: 120,
                            damping: 20,
                            mass: 1,
                            delay: index * 0.05,
                          },
                        },
                      }}
                    >
                      {size.label}
                    </motion.button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </motion.div>
      </div>

      <Cart isOpen={isCartOpen} onClose={() => setIsCartOpen(false)} />
    </motion.div>
  );
}
