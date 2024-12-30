import { products } from '@/lib/products';
import PDP from './pdp';

export async function generateStaticParams() {
  return products.map((product) => ({
    slug: product.id,
  }));
}

export default async function ProductPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const slug = (await params).slug;
  return <PDP slug={slug} />;
}
