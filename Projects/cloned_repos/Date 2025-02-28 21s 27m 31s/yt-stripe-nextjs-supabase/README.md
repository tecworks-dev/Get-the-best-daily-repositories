# Next.js + Stripe + Supabase Production-Ready Template

A production-ready Next.js template featuring authentication, dark mode support, Stripe integration, and a clean, modern UI. Built with TypeScript and Tailwind CSS.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)
![Tailwind](https://img.shields.io/badge/Tailwind-3.0-38B2AC)

Full Video Guide: https://www.youtube.com/watch?v=ad1BxZufer8&list=PLE9hy4A7ZTmpGq7GHf5tgGFWh2277AeDR&index=8

## ✨ Features

- 🔐 Authentication with Supabase
- 💳 Stripe payment integration
- 🌓 Dark mode support
- 📱 Responsive design
- 🎨 Tailwind CSS styling
- 🔄 Framer Motion animations
- 🛡️ TypeScript support
- 📊 Error boundary implementation
- 🔍 SEO optimized

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- A Supabase account
- A Stripe account
- A Google Cloud Platform account

### Installation and Setup

1. Clone the template:

```bash
git clone https://github.com/ShenSeanChen/yt-stripe-nextjs-supabase my-full-stack-app
cd my-full-stack-app
rm -rf .git
git init
git add .
git commit -m "Initial commit"
# git remote add origin https://github.com/ShenSeanChen/my-full-stack-app.git
git remote add origin https://github.com/USE_YOUR_OWN_GITHUB_NAME/my-full-stack-app.git
git push -u origin main
```

2. Install dependencies:
```bash
npm install
```
or
```bash
yarn install
```

3. Create .env.local with all variables from .env.example
```
NEXT_PUBLIC_APP_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8080

# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# OpenAI Configuration (you'll need to add your key)
OPENAI_API_KEY=

# Stripe Configuration
# NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_
NEXT_PUBLIC_STRIPE_BUTTON_ID=buy_btn_
# STRIPE_SECRET_KEY=sk_test_
STRIPE_SECRET_KEY=sk_live_
# STRIPE_WEBHOOK_SECRET=whsec_
STRIPE_WEBHOOK_SECRET=whsec_

# ANALYTICS
NEXT_PUBLIC_POSTHOG_KEY=
NEXT_PUBLIC_POSTHOG_HOST=https://app.posthog.com
```

4. Set up Google Cloud Platform (GCP):
   - Create new OAuth 2.0 credentials in GCP Console
   - Configure authorized JavaScript origins
   - Configure redirect URIs
   - Save the Client ID and Client Secret

5. Configure Supabase:

   a. Get API Keys (Project Settings > API):
      - Project URL → NEXT_PUBLIC_SUPABASE_URL
      - Anon Public Key → NEXT_PUBLIC_SUPABASE_ANON_KEY
      - Service Role Secret → SUPABASE_SERVICE_ROLE_KEY
   
   b. Set up Authentication:
      - Go to Authentication > Providers > Google
      - Add your GCP Client ID and Client Secret
      - Update Site URL and Redirect URLs
   
   c. Database Setup:
      - Enable Row Level Security (RLS) for all tables
      - Create policies for authenticated users and service roles
      - Create the following trigger function:

      ```sql
      CREATE OR REPLACE FUNCTION public.handle_new_user()
      RETURNS trigger AS $$
      BEGIN
        INSERT INTO public.users (id, email, created_at, updated_at, is_deleted)
        VALUES (NEW.id, NEW.email, NOW(), NOW(), FALSE);
        
        INSERT INTO public.user_preferences (user_id, has_completed_onboarding)
        VALUES (NEW.id, FALSE);
        
        INSERT INTO public.user_trials (user_id, trial_start_time, trial_end_time)
        VALUES (NEW.id, NOW(), NOW() + INTERVAL '48 hours');
        
        RETURN NEW;
      END;
      $$ LANGUAGE plpgsql SECURITY DEFINER;

      CREATE TRIGGER on_auth_user_created
        AFTER INSERT ON auth.users
        FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
      ```

6. Set up Stripe:
   a. Create a live account and configure:
      - Create product in Product Catalog
      - Create promotional coupon codes
      - Set up Payment Link with trial period
   
   b. Get required keys:
      - Publishable Key → NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY
      - Secret Key → STRIPE_SECRET_KEY
      - Buy Button ID → NEXT_PUBLIC_STRIPE_BUTTON_ID
   
   c. Configure webhooks:
      - Add endpoint: your_url/api/stripe/webhook
      - Subscribe to events: customer.subscription.*, checkout.session.*, invoice.*, payment_intent.*
      - Copy Signing Secret → STRIPE_WEBHOOK_SECRET

7. Start the development server:
```bash
npm run dev
```
or
```bash
yarn dev
```

8. Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📖 Project Structure

```
├── app/                  # Next.js 14 app directory
│   ├── api/              # API routes
│   │   ├── stripe/       # Stripe payment endpoints
│   │   └── user/         # User API endpoints
│   ├── auth/             # Auth-related pages
│   │   ├── callback/     # handle auth callback
│   ├── dashboard/        # Dashboard pages
│   ├── pay/              # Payment pages
│   ├── profile/          # User profile pages
│   ├── reset-password/   # Reset password pages
│   ├── update-password/  # Update password pages
│   ├── verify-email/     # Verify email pages
│   ├── layout.tsx        # Root layout
│   └── page.tsx          # Home page
├── components/           # Reusable components
├── contexts/             # React contexts
├── hooks/                # Custom React hooks
├── utils/                # Utility functions
├── types/                # TypeScript type definitions
├── public/               # Static assets
└── styles/               # Global styles
```

## 🛠️ Built With

- [Next.js](https://nextjs.org/) - React framework
- [TypeScript](https://www.typescriptlang.org/) - Type safety
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [Supabase](https://supabase.com/) - Authentication & Database
- [Stripe](https://stripe.com/) - Payments
- [Framer Motion](https://www.framer.com/motion/) - Animations

## 🔧 Configuration

### Tailwind Configuration

The template includes a custom Tailwind configuration with:
- Custom colors
- Dark mode support
- Extended theme options
- Custom animations

### Authentication

Authentication is handled through Supabase with support for:
- Email/Password
- Google OAuth
- Magic Links
- Password Reset

### Payment Integration

Stripe integration includes:
- Subscription management
- Trial periods
- Webhook handling
- Payment status tracking

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Next.js team for the amazing framework
- Vercel for the deployment platform
- Tailwind CSS team for the utility-first CSS framework
- Supabase team for the backend platform
- Stripe team for the payment infrastructure

## 📫 Contact

X - [@ShenSeanChen](https://x.com/ShenSeanChen)

YouTube - [@SeanTechStories](https://www.youtube.com/@SeanTechStories)

Discord - [@Sean's Stories](https://discord.gg/TKKPzZheua)

Instagram - [@SeanTechStories](https://www.instagram.com/sean_tech_stories )

Project Link: [https://github.com/ShenSeanChen/yt-stripe-nextjs-supabase](https://github.com/ShenSeanChen/yt-stripe-nextjs-supabase)

## 🚀 Deploy

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js).

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/your-repo-name)

---

Made with 🔥 by [ShenSeanChen](https://github.com/ShenSeanChen)
