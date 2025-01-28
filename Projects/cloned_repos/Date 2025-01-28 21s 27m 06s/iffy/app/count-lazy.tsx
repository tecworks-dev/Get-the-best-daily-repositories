"use client";

import dynamic from "next/dynamic";

// The ssr: false option only works from a client component,
// hence the need for this wrapper
// https://nextjs.org/docs/app/building-your-application/optimizing/lazy-loading#skipping-ssr
const Count = dynamic(() => import("./count").then((mod) => mod.Count), {
  ssr: false,
  loading: () => <span>...</span>,
});

export function CountLazy({ count, countAt, ratePerHour }: { count: number; countAt: Date; ratePerHour: number }) {
  return <Count count={count} countAt={countAt} ratePerHour={ratePerHour} />;
}
