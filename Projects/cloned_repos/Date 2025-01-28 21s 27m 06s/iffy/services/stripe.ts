import Stripe from "stripe";

export async function getPaymentsAndPayouts(stripeApiKey: string, stripeAccountId: string) {
  if (!stripeApiKey) {
    throw new Error("Stripe API key not provided");
  }

  const stripe = new Stripe(stripeApiKey);
  const account = await stripe.accounts.retrieve(stripeAccountId);

  return {
    payments: account.charges_enabled,
    payouts: account.payouts_enabled,
  };
}

export async function pausePaymentsAndPayouts(stripeApiKey: string, stripeAccountId: string) {
  if (!stripeApiKey) {
    throw new Error("Stripe API key not provided");
  }

  const stripe = new Stripe(stripeApiKey);

  await stripe.accounts.update(stripeAccountId, {
    // @ts-ignore preview feature
    risk_controls: {
      payouts: {
        pause_requested: true,
      },
      charges: {
        pause_requested: true,
      },
    },
  });
}

export async function resumePaymentsAndPayouts(stripeApiKey: string, stripeAccountId: string) {
  if (!stripeApiKey) {
    throw new Error("Stripe API key not provided");
  }

  const stripe = new Stripe(stripeApiKey);

  await stripe.accounts.update(stripeAccountId, {
    // @ts-ignore preview feature
    risk_controls: {
      payouts: {
        pause_requested: false,
      },
      charges: {
        pause_requested: false,
      },
    },
  });
}
