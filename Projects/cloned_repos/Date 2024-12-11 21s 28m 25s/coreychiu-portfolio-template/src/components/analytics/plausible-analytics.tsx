"use client";

import Script from "next/script";

const plausibleUrl = process.env.NEXT_PUBLIC_PLAUSIBLE_URL!;
const plausibleSrc = process.env.NEXT_PUBLIC_PLAUSIBLE_SRC!;


export function PlausibleAnalytics() {
  if (process.env.NODE_ENV !== "production") {
    return null;
  }

  return (
    <Script
      defer
      type="text/javascript"
      data-domain={plausibleUrl}
      src={plausibleSrc}
    />
  );
}

export function usePlausibleAnalytics() {
  const trackEvent = (event: string, data?: Record<string, unknown>) => {
    if (typeof window === "undefined" || !(window as any).plausible) {
      return;
    }

    (window as any).plausible(event, {
      props: data,
    });
  };

  return {
    trackEvent,
  };
}
