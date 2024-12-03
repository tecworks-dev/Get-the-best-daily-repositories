import { NextRequest } from "next/server";
import Stripe from "stripe";
import { toDate as zonedTimeToUtc, toZonedTime, format } from 'date-fns-tz'
import { startOfMonth, endOfMonth } from 'date-fns'

interface PaymentData {
  id: string;
  amount: number;
  currency: string;
  created: number;
  paymentType: 'one_time' | 'subscription';
  subscriptionInterval?: 'month' | 'year';
  subscriptionType?: 'new' | 'renewal';
  status: string;
}

interface StripeDataResponse {
  totalOrders: number;
  totalAmount: number;
  currency: string;
  payments: PaymentData[];
  accountName: string;
  upcomingRenewals: UpcomingRenewal[];
}

interface UpcomingRenewal {
  customerId: string;
  amount: number;
  currency: string;
  renewalDate: number;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const password = searchParams.get('password');
  
  if (!password || password !== process.env.STRIPE_VIEWER_PASSWORD) {
    return new Response('Unauthorized', { status: 401 });
  }

  const headers = new Headers({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  try {
    const stripeKeys = Object.entries(process.env)
      .filter(([key]) => key.startsWith('STRIPE_SECRET_KEY_'))
      .map(([_, value]) => value);

    const stream = new ReadableStream({
      async start(controller) {
        try {
          controller.enqueue(
            `data: ${JSON.stringify({ type: 'total', count: stripeKeys.length })}\n\n`
          );

          const { searchParams } = new URL(request.url);
          const date = searchParams.get('date');
          const timezone = searchParams.get('timezone') || 'UTC';

          if (!date) {
            throw new Error('date is required');
          }

          const isMonthQuery = date.length === 7;
          
          let startTimestamp, endTimestamp;
          
          if (isMonthQuery) {
            const [year, month] = date.split('-');
            const zonedStartDate = toZonedTime(
              new Date(`${year}-${month}-01T00:00:00`),
              timezone
            );
            
            const monthStart = startOfMonth(zonedStartDate);
            const monthEnd = endOfMonth(zonedStartDate);
            
            startTimestamp = Math.floor(zonedTimeToUtc(monthStart, { timeZone: timezone }).getTime() / 1000);
            endTimestamp = Math.floor(zonedTimeToUtc(monthEnd, { timeZone: timezone }).setHours(23, 59, 59, 999) / 1000);

            console.log({
              queryDate: date,
              timezone,
              zonedStartDate: format(zonedStartDate, 'yyyy-MM-dd HH:mm:ssXXX', { timeZone: timezone }),
              monthStart: format(monthStart, 'yyyy-MM-dd HH:mm:ssXXX', { timeZone: timezone }),
              monthEnd: format(monthEnd, 'yyyy-MM-dd HH:mm:ssXXX', { timeZone: timezone }),
              startTimestamp: new Date(startTimestamp * 1000).toISOString(),
              endTimestamp: new Date(endTimestamp * 1000).toISOString(),
            });
          } else {
            const zonedStartDate = toZonedTime(new Date(`${date}T00:00:00`), timezone);
            const zonedEndDate = toZonedTime(new Date(`${date}T23:59:59.999`), timezone);

            startTimestamp = Math.floor(zonedTimeToUtc(zonedStartDate, { timeZone: timezone }).getTime() / 1000);
            endTimestamp = Math.floor(zonedTimeToUtc(zonedEndDate, { timeZone: timezone }).getTime() / 1000);

            console.log({
              queryDate: date,
              timezone,
              zonedStartDate: format(zonedStartDate, 'yyyy-MM-dd HH:mm:ssXXX', { timeZone: timezone }),
              zonedEndDate: format(zonedEndDate, 'yyyy-MM-dd HH:mm:ssXXX', { timeZone: timezone }),
              startTimestamp: new Date(startTimestamp * 1000).toISOString(),
              endTimestamp: new Date(endTimestamp * 1000).toISOString()
            });
          }

          for (let i = 0; i < stripeKeys.length; i++) {
            const stripe = new Stripe(stripeKeys[i]!);
            
            const merchantData = await fetchMerchantData(
              stripe, 
              startTimestamp, 
              endTimestamp,
              timezone
            );
            
            controller.enqueue(
              `data: ${JSON.stringify({
                type: 'data',
                current: i + 1,
                total: stripeKeys.length,
                data: {
                  merchantId: `merchant_${i + 1}`,
                  ...merchantData,
                }
              })}\n\n`
            );
          }

          controller.close();
        } catch (error) {
          controller.error(error);
        }
      }
    });

    return new Response(stream, { headers });
  } catch (error) {
    console.error('Error processing Stripe data:', error);
    return new Response(JSON.stringify({ error: 'Internal Server Error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

async function fetchMerchantData(
  stripe: Stripe,
  startTimestamp: number,
  endTimestamp: number,
  timezone: string
): Promise<StripeDataResponse> {
  const account = await stripe.accounts.retrieve();
  const accountName = account.settings?.dashboard?.display_name || account.business_profile?.name || 'Unknown';

  console.log(`\n=== Fetching data for ${accountName} ===`);
  console.log(`Time range: ${new Date(startTimestamp * 1000).toISOString()} to ${new Date(endTimestamp * 1000).toISOString()}`);

  const allCharges: Stripe.Charge[] = [];
  let hasMoreCharges = true;
  let startingAfterCharge: string | undefined = undefined;

  while (hasMoreCharges) {
    const charges:any = await stripe.charges.list({
      created: {
        gte: startTimestamp,
        lte: endTimestamp,
      },
      limit: 100,
      starting_after: startingAfterCharge,
      expand: ['data.invoice', 'data.invoice.subscription'],
    });

    allCharges.push(...charges.data);
    
    if (charges.has_more && charges.data.length > 0) {
      startingAfterCharge = charges.data[charges.data.length - 1].id;
    } else {
      hasMoreCharges = false;
    }
  }

  const allPaymentIntents: Stripe.PaymentIntent[] = [];
  let hasMorePI = true;
  let startingAfterPI: string | undefined = undefined;

  while (hasMorePI) {
    const paymentIntents:any = await stripe.paymentIntents.list({
      created: {
        gte: startTimestamp,
        lte: endTimestamp,
      },
      limit: 100,
      starting_after: startingAfterPI,
      expand: ['data.invoice', 'data.invoice.subscription'],
    });

    console.log(`\nFetched ${paymentIntents.data.length} PaymentIntents:`);
    paymentIntents.data.forEach((pi: any) => {
      console.log(`PI ${pi.id}:
        - created: ${new Date(pi.created * 1000).toISOString()}
        - status: ${pi.status}
        - amount: ${pi.amount}
        - latest_charge: ${pi.latest_charge}
      `);
    });

    allPaymentIntents.push(...paymentIntents.data);
    
    if (paymentIntents.has_more && paymentIntents.data.length > 0) {
      startingAfterPI = paymentIntents.data[paymentIntents.data.length - 1].id;
      hasMorePI = true;
    } else {
      hasMorePI = false;
    }
  }

  const missingPaymentIntents = new Set(
    allCharges
      .filter((charge: any) => 
        charge.status === 'succeeded' && 
        charge.payment_intent && 
        !allPaymentIntents.some(pi => pi.id === charge.payment_intent)
      )
      .map((charge: any) => charge.payment_intent as string)
  );

  if (missingPaymentIntents.size > 0) {
    console.log(`\nFetching ${missingPaymentIntents.size} missing PaymentIntents`);
    for (const piId of Array.from(missingPaymentIntents)) {
      try {
        const pi = await stripe.paymentIntents.retrieve(piId, {
          expand: ['invoice', 'invoice.subscription'],
        });
        console.log(`Retrieved missing PI ${pi.id}`);
        allPaymentIntents.push(pi);
      } catch (error) {
        console.error(`Failed to retrieve PI ${piId}:`, error);
      }
    }
  }

  console.log('\nFiltering charges:');
  const validCharges = allCharges.filter(charge => {
    const isValid = charge.status === 'succeeded' && 
      !charge.refunded &&
      !charge.payment_intent && 
      !allPaymentIntents.some(pi => pi.latest_charge === charge.id);
    
    console.log(`Charge ${charge.id}: ${isValid ? 'VALID' : 'EXCLUDED'} (status=${charge.status}, refunded=${charge.refunded}, pi=${charge.payment_intent})`);
    
    return isValid;
  });

  const processedPayments: any = [
    ...allPaymentIntents
      .filter(pi => pi.status === 'succeeded')
      .map(pi => ({
        id: pi.id,
        amount: pi.amount,
        currency: pi.currency,
        created: pi.created,
        status: pi.status,
        paymentType: pi.invoice ? 'subscription' : 'one_time',
        subscriptionInterval: pi.invoice ? 
          ((pi.invoice as Stripe.Invoice).subscription as Stripe.Subscription)?.items?.data[0]?.price?.recurring?.interval as 'month' | 'year' || 'month' 
          : undefined,
        subscriptionType: pi.invoice ? 
          ((pi.invoice as Stripe.Invoice).billing_reason === 'subscription_create' ? 'new' : 'renewal')
          : undefined,
      })),
    ...validCharges
      .map(charge => ({
        id: charge.id,
        amount: charge.amount,
        currency: charge.currency,
        created: charge.created,
        status: charge.status,
        paymentType: charge.invoice ? 'subscription' : 'one_time',
        subscriptionInterval: charge.invoice ?
          ((charge.invoice as Stripe.Invoice).subscription as Stripe.Subscription)?.items?.data[0]?.price?.recurring?.interval as 'month' | 'year' || 'month'
          : undefined,
        subscriptionType: charge.invoice ?
          ((charge.invoice as Stripe.Invoice).billing_reason === 'subscription_create' ? 'new' : 'renewal')
          : undefined,
      }))
  ];

  console.log('Processed payments:');
  processedPayments.forEach((payment: any) => {
    console.log(`- ${payment.id}: ${payment.paymentType} ${payment.amount} ${payment.currency}`);
  });

  // Fetch subscriptions
  const allSubscriptions: Stripe.Subscription[] = [];
  let hasMoreSubscriptions = true;
  let startingAfterSubscription: string | undefined = undefined;

  while (hasMoreSubscriptions) {
    const subscriptions: any = await stripe.subscriptions.list({
      status: 'active',
      limit: 100,
      starting_after: startingAfterSubscription,
    });

    allSubscriptions.push(...subscriptions.data);

    if (subscriptions.has_more && subscriptions.data.length > 0) {
      startingAfterSubscription = subscriptions.data[subscriptions.data.length - 1].id;
    } else {
      hasMoreSubscriptions = false;
    }
  }

  // Filter for upcoming renewals
  const upcomingRenewals = allSubscriptions
    .filter(subscription => 
      !subscription.cancel_at_period_end && 
      subscription.status === 'active' &&
      subscription.current_period_end >= startTimestamp &&
      subscription.current_period_end <= endTimestamp
    )
    .map(subscription => ({
      customerId: subscription.customer as string,
      amount: subscription.items.data[0].price.unit_amount || 0,
      currency: subscription.items.data[0].price.currency,
      renewalDate: subscription.current_period_end,
    }));

  return {
    totalOrders: processedPayments.length,
    totalAmount: processedPayments.reduce((sum: number, payment: any) => sum + payment.amount, 0),
    currency: processedPayments[0]?.currency || 'usd',
    payments: processedPayments,
    accountName,
    upcomingRenewals,
  };
} 