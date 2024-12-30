import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Terms - NEXYZY',
  description: 'Terms and conditions for NEXYZY.',
};

export default function TermsPage() {
  return (
    <div className="mt-12 space-y-12 font-mono max-w-[700px] mx-auto">
      <h1 className="text-xl">TERMS</h1>

      <section className="space-y-4">
        <h2 className="text-lg">TERMS OF SERVICE</h2>
        <p className="leading-relaxed">
          By accessing and placing an order with NEXYZY, you confirm that you
          are in agreement with and bound by the terms and conditions contained
          herein.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg">PRICING</h2>
        <p className="leading-relaxed">
          All prices are final and non-negotiable. Prices are listed in USD and
          do not include taxes or shipping costs, which will be calculated at
          checkout.
        </p>
      </section>
    </div>
  );
}
