'use client';

import { useState, useEffect, useCallback, useTransition } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { products, Product } from '@/lib/products';
import { Header } from '@/components/header';
import { AddToCart } from '@/components/add-to-cart';
import { ProductImage } from '@/components/product-image';

export default function Page() {
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [_, startTransition] = useTransition();

  const handleProductClick = (product: Product) => {
    startTransition(() => {
      setSelectedProduct(product);
      window.history.pushState(null, '', `/p/${product.id}`);
    });
  };

  const handleBack = useCallback(() => {
    startTransition(() => {
      setSelectedProduct(null);
      window.history.pushState(null, '', '/');
    });
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (selectedProduct) {
        if (event.key === 'Escape') {
          handleBack();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedProduct, handleBack]);

  useEffect(() => {
    const handlePopState = () => {
      const productId = window.location.pathname.split('/').pop();
      if (productId && productId !== '') {
        const product = products.find((p) => p.id === productId);
        if (product) {
          setSelectedProduct(product);
        } else {
          setSelectedProduct(null);
        }
      } else {
        setSelectedProduct(null);
      }
    };

    window.addEventListener('popstate', handlePopState);

    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  return (
    <div className="flex flex-col min-h-screen">
      <Header isBackVisible={!!selectedProduct} onBack={handleBack} />
      <main className="flex-grow relative pt-12">
        <motion.div
          className="grid grid-cols-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-9 gap-x-5 gap-y-12 pb-8"
          animate={{ opacity: selectedProduct ? 0 : 1 }}
          transition={{ duration: 0.3 }}
        >
          {products.map((product) => (
            <div
              key={product.id}
              className="group cursor-pointer"
              onClick={() => handleProductClick(product)}
            >
              <ProductImage
                product={product}
                layoutId={`product-image-${product.id}`}
              />
              <p className="font-medium text-center font-mono uppercase">
                {product.id.split('-').slice(0, -1).join('-')}
              </p>
            </div>
          ))}
        </motion.div>

        <AnimatePresence>
          {selectedProduct && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 flex flex-col items-center justify-between bg-white bg-opacity-90"
              style={{
                top: '0',
                height:
                  'calc(100vh - 80px - env(safe-area-inset-top) - env(safe-area-inset-bottom))',
                paddingTop: 'calc(20px + env(safe-area-inset-top))',
                paddingBottom: '0',
              }}
            >
              <div className="w-full max-w-4xl mx-auto flex-grow flex flex-col items-center justify-center p-4">
                <ProductImage
                  product={selectedProduct}
                  maxWidth="100%"
                  maxHeight="calc(100vh - 250px - env(safe-area-inset-top) - env(safe-area-inset-bottom))"
                  className="w-full"
                  layoutId={`product-image-${selectedProduct.id}`}
                />
              </div>

              <motion.div
                className="w-full max-w-md mx-auto p-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.3 }}
              >
                <AddToCart product={selectedProduct} />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
