import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import Stripe from 'stripe';
import { supabaseAdmin } from '@/utils/supabase-admin';
import { withCors } from '@/utils/cors';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!);

export const DELETE = withCors(async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');

    if (!userId) {
      return NextResponse.json({ error: 'User ID is required' }, { status: 400 });
    }

    console.log('Starting account soft-deletion for user:', userId);

    // 1. Cancel Stripe subscriptions if they exist
    const { data: subscriptionsData, error: subError } = await supabaseAdmin
      .from('subscriptions')
      .select('stripe_subscription_id, status')
      .eq('user_id', userId);

    if (subError) {
      console.error('Subscription fetch error:', subError);
    } else if (subscriptionsData) {
      for (const sub of subscriptionsData) {
        if (sub.stripe_subscription_id && (sub.status === 'active' || sub.status === 'trialing')) {
          try {
            await stripe.subscriptions.cancel(sub.stripe_subscription_id);
            console.log('Stripe subscription cancelled:', sub.stripe_subscription_id);
          } catch (stripeError) {
            console.error('Stripe cancellation error:', stripeError);
          }
        }
      }
    }

    // 2. Soft delete the profile
    const { error: profileError } = await supabaseAdmin
      .from('users')
      .update({ 
        deleted_at: new Date().toISOString(),
        is_deleted: true
      })
      .eq('id', userId);

    if (profileError) {
      console.error('Profile update error:', profileError);
      return NextResponse.json(
        { error: 'Failed to update profile', details: profileError },
        { status: 500 }
      );
    }

    // 3. Mark subscriptions as canceled
    const { error: subscriptionUpdateError } = await supabaseAdmin
      .from('subscriptions')
      .update({
        deleted_at: new Date().toISOString(),
        status: 'canceled'
      })
      .eq('user_id', userId);

    if (subscriptionUpdateError) {
      console.error('Subscription update error:', subscriptionUpdateError);
    }

    console.log('Account soft-deletion completed successfully');
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error in account soft-deletion:', error);
    return NextResponse.json(
      { 
        error: 'Failed to process account deletion', 
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}); 