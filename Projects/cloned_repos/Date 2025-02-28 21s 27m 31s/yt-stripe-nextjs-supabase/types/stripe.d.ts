// The .d.ts extension is a convention that tells TypeScript that this file is purely for type declarations and shouldn't be compiled into JavaScript.

// Import the Stripe namespace
import Stripe from 'stripe';

// Extend the JSX namespace for the Stripe Buy Button
declare namespace JSX {
  interface IntrinsicElements {
    'stripe-buy-button': {
      'buy-button-id': string;
      'publishable-key': string;
    }
  }
}

// Export Stripe types if needed elsewhere
export type StripeCheckoutSession = Stripe.Checkout.Session;
export type StripeEvent = Stripe.Event; 