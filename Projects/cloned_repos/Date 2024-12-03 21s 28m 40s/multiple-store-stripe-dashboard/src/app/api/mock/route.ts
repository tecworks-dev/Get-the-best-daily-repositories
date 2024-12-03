import { NextRequest } from "next/server";

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

function generateMockData(): StripeDataResponse[] {
  const merchants = [
    {
      merchantId: 'merchant_1',
      accountName: 'Premium Software Inc.',
      currency: 'usd',
    },
    {
      merchantId: 'merchant_2',
      accountName: 'Digital Tools Pro',
      currency: 'usd',
    },
    {
      merchantId: 'merchant_3',
      accountName: 'Cloud Services Plus',
      currency: 'usd',
    }
  ];

  const startDate = new Date('2024-11-01T00:00:00Z').getTime() / 1000;
  const endDate = new Date('2024-11-30T23:59:59Z').getTime() / 1000;

  return merchants.map((merchant, index) => {
    const payments: PaymentData[] = [];
    const daysInMonth = 30;
    
    const traits = {
      oneTimePrice: [49, 99, 199][index],
      monthlyPrice: [29, 49, 79][index],
      yearlyPrice: [290, 490, 790][index],
      oneTimeRatio: [0.4, 0.3, 0.2][index],
      subscriptionRatio: [0.6, 0.7, 0.8][index],
      newVsRenewalRatio: [0.3, 0.4, 0.5][index],
      monthlyVsYearlyRatio: [0.7, 0.6, 0.5][index],
    };

    const baseOrdersPerDay = [8, 12, 15][index];

    for (let day = 1; day <= daysInMonth; day++) {
      const timestamp = startDate + (day - 1) * 86400;
      const dailyMultiplier = 1 + Math.sin(day * 0.5) * 0.3;
      const ordersToday = Math.floor(baseOrdersPerDay * dailyMultiplier);

      for (let i = 0; i < ordersToday; i++) {
        const isSubscription = Math.random() < traits.subscriptionRatio;
        
        if (isSubscription) {
          const isNewSub = Math.random() < traits.newVsRenewalRatio;
          const isMonthly = Math.random() < traits.monthlyVsYearlyRatio;
          
          payments.push({
            id: `pi_${merchant.merchantId}_${day}_${i}`,
            amount: isMonthly ? traits.monthlyPrice * 100 : traits.yearlyPrice * 100,
            currency: merchant.currency,
            created: timestamp + i * 300,
            paymentType: 'subscription',
            subscriptionInterval: isMonthly ? 'month' : 'year',
            subscriptionType: isNewSub ? 'new' : 'renewal',
            status: 'succeeded'
          });
        } else {
          payments.push({
            id: `ch_${merchant.merchantId}_${day}_${i}`,
            amount: traits.oneTimePrice * 100,
            currency: merchant.currency,
            created: timestamp + i * 300,
            paymentType: 'one_time',
            status: 'succeeded'
          });
        }
      }
    }

    const upcomingRenewals: UpcomingRenewal[] = Array.from({ length: Math.floor(Math.random() * 10) + 5 }, (_, i) => ({
      customerId: `cus_${merchant.merchantId}_${i}`,
      amount: traits.monthlyPrice * 100,
      currency: merchant.currency,
      renewalDate: endDate + (i + 1) * 86400
    }));

    return {
      totalOrders: payments.length,
      totalAmount: payments.reduce((sum, p) => sum + p.amount, 0),
      currency: merchant.currency,
      payments,
      accountName: merchant.accountName,
      upcomingRenewals
    };
  });
}

export async function GET(request: NextRequest) {
  const mockData = generateMockData();
  return new Response(JSON.stringify(mockData), {
    headers: { 'Content-Type': 'application/json' },
  });
}