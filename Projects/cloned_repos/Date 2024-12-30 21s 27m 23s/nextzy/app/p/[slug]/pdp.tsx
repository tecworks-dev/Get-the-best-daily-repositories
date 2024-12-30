'use client';

import { motion } from 'motion/react';
import { notFound, redirect } from 'next/navigation';
import { getProductById } from '@/lib/products';
import { AddToCart } from '@/components/add-to-cart';
import { Header } from '@/components/header';
import { ProductImage } from '@/components/product-image';

export default function PDP({ slug }: { slug: string }) {
  const product = getProductById(slug);

  if (!product) {
    notFound();
  }

  const handleBack = () => {
    redirect('/');
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Header isBackVisible={true} onBack={handleBack} />
      <main className="flex flex-col items-center justify-between pt-[20px]">
        <div className="w-full max-w-4xl mx-auto flex-grow flex flex-col items-center justify-center p-4">
          <ProductImage
            product={product}
            maxWidth="100%"
            maxHeight="calc(100vh - 250px - env(safe-area-inset-top) - env(safe-area-inset-bottom))"
            className="w-full"
          />
        </div>

        <motion.div
          className="w-full max-w-md mx-auto mt-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
        >
          <AddToCart product={product} />
        </motion.div>
      </main>
    </div>
  );
}
