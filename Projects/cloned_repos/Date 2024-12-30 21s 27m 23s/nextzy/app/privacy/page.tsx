import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Privacy - NEXYZY',
  description: 'Privacy policy for NEXYZY.',
};

export default function PrivacyPage() {
  return (
    <div className="mt-12 space-y-12 font-mono max-w-[700px] mx-auto">
      <h1 className="text-xl">PRIVACY</h1>

      <section className="space-y-4">
        <h2 className="text-lg">DATA COLLECTION</h2>
        <p className="leading-relaxed">
          We collect minimal personal information necessary for order processing
          and delivery. This includes your name, shipping address, and contact
          information.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg">DATA USAGE</h2>
        <p className="leading-relaxed">
          Your information is used solely for order fulfillment and shipping
          purposes. We do not share or sell your data to third parties.
        </p>
      </section>
    </div>
  );
}
