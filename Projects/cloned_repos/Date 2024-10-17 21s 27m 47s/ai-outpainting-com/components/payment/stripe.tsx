'use client'
import { Elements, ElementsConsumer, PaymentElement } from '@stripe/react-stripe-js'
import { Button } from '@nextui-org/button'
import { loadStripe } from '@stripe/stripe-js'
import { t } from '@lingui/macro'
// fix
export const _VERSION = true

function StripeForm({ stripe, elements }: { stripe: any; elements: any }) {
  async function handleSubmit(e: any) {
    e.preventDefault()
    if (!stripe || !elements) {
      return
    }
    const res = await stripe.confirmPayment({
      elements,
      confirmParams: {
        return_url: ''
      }
    })
  }

  return (
    <form onSubmit={handleSubmit}>
      <PaymentElement />
      <div className="mt-5">
        <Button
          type="submit"
          color="primary"
          size="lg"
          fullWidth
          disabled={!stripe}
          className={'bg-black'}
        >
          {t`Go to Payment`}
        </Button>
      </div>
    </form>
  )
}

export default function PaymentStripeClient({
                                              locale,
                                              clientSecret
                                            }: {
  clientSecret: string
  locale: string
}) {
  const stripe = loadStripe(process.env.UE_STRIPE_PK ?? '')
  const options = {
    clientSecret,
    locale: locale ?? 'en',
    appearance: {
      theme: 'flat'
    }
  }
  return (
    <Elements stripe={stripe} options={options as any}>
      <ElementsConsumer>
        {({ stripe, elements }) => (
          <StripeForm stripe={stripe} elements={elements} />
        )}
      </ElementsConsumer>
    </Elements>
  )
}
