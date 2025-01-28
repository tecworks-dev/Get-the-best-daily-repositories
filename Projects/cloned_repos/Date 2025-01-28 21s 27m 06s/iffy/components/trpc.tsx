"use client";

import { trpc } from "@/lib/trpc";
import { getAbsoluteUrl } from "@/lib/url";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { getFetch, httpLink, loggerLink } from "@trpc/react-query";
import { useState } from "react";
import superjson from "superjson";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 5 * 1000 } },
});

export default function TRPCProvider({ children }: { children: React.ReactNode }) {
  const [trpcClient] = useState(() =>
    trpc.createClient({
      links: [
        loggerLink({
          enabled: () => true,
        }),
        // we use httpLink instead of httpBatchLink
        // since we only have a few queries per page
        // so we don't get much benefit from batching
        // additionally, it's helpful to allow slow queries to be fetched separately
        httpLink({
          url: getAbsoluteUrl("/api/trpc"),
          fetch: async (input, init?) => {
            const fetch = getFetch();
            return fetch(input, {
              ...init,
              credentials: "include",
            });
          },
          transformer: superjson,
        }),
      ],
    }),
  );

  return (
    <trpc.Provider client={trpcClient} queryClient={queryClient}>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </trpc.Provider>
  );
}
