import Image from 'next/image';
import { motion } from 'motion/react';
import { Product } from '@/lib/products';

interface ProductImageProps {
  product: Product;
  priority?: boolean;
  maxWidth?: string;
  maxHeight?: string;
  className?: string;
  layoutId?: string;
}

export function ProductImage({
  product,
  maxWidth = '100%',
  maxHeight = 'none',
  className = '',
  layoutId,
}: ProductImageProps) {
  return (
    <motion.div
      className={`relative mb-1 ${className}`}
      style={{
        width: '100%',
        maxWidth,
        maxHeight,
        aspectRatio: '1',
        overflow: 'hidden',
      }}
      layoutId={layoutId}
    >
      <Image
        src={product.image}
        alt={product.name}
        fill
        sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
        className="object-contain transition-opacity duration-200 aspect-square"
        loading="eager"
        decoding="sync"
      />
    </motion.div>
  );
}
