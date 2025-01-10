"use client"


import type { MaterializedQueryResult, MDConnection, SafeQueryResult } from "@motherduck/wasm-client";
import 'core-js/actual/promise/with-resolvers';
import { createContext, useContext, useEffect, useMemo, useRef } from "react";
import initMotherDuckConnection from "../functions/initMotherDuckConnection";

interface MotherDuckContextValue {
  evaluateQuery: (query: string) => Promise<MaterializedQueryResult>;
  safeEvaluateQuery: (query: string) => Promise<SafeQueryResult<MaterializedQueryResult>>;
}

export const MotherDuckContext = createContext<MotherDuckContextValue | null>(null);

export function MotherDuckClientProvider({ children, database }: { children: React.ReactNode, database?: string },) {
  const connectionRef = useRef<PromiseWithResolvers<MDConnection | undefined>>();

  if (connectionRef.current === undefined) {
    connectionRef.current = Promise.withResolvers<MDConnection | undefined>();
  }

  const evaluateQuery = async (query: string): Promise<MaterializedQueryResult> => {
    if (!connectionRef.current) {
      throw new Error('MotherDuck connection ref is falsy')
    }

    const connection = await connectionRef.current.promise;

    if (!connection) {
      throw new Error('No MotherDuck connection available');
    }

    return connection.evaluateQuery(query);
  };

  const safeEvaluateQuery = async (query: string): Promise<SafeQueryResult<MaterializedQueryResult>> => {
    if (!connectionRef.current) {
      throw new Error('MotherDuck connection ref is falsy')
    }

    const connection = await connectionRef.current.promise;

    if (!connection) {
      throw new Error('No MotherDuck connection available');
    }

    return connection.safeEvaluateQuery(query);
  };

  useEffect(() => {
    const initializeConnection = async () => {
      try {
        // read only public token
        const mdToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InJoeXNAcmh5c3N1bGxpdmFuLmNvbSIsInNlc3Npb24iOiJyaHlzLnJoeXNzdWxsaXZhbi5jb20iLCJwYXQiOiJhU1ZUSlhOZmMwbEIyZ0Yzc2lyaDc3WDhDalhTbWpjb0pHWFhRWm5lSmw4IiwidXNlcklkIjoiZTY2M2QwMWItMTA3MS00NTJiLWI4YzgtMDNkNjYzYWM2YWI4IiwiaXNzIjoibWRfcGF0IiwicmVhZE9ubHkiOnRydWUsInRva2VuVHlwZSI6InJlYWRfc2NhbGluZyIsImlhdCI6MTczNjQ1NjYyNH0.DDWrhfebXrZnTbG2dRurXWKvhVG1JtIkQYELwBdOLEM'
        const result = initMotherDuckConnection(mdToken, database);
        if (connectionRef.current) {
          connectionRef.current.resolve(result);
        }
      } catch (error) {
        console.error(error);
      }
    };
    initializeConnection();
  }, []);

  const value = useMemo(() => ({
    evaluateQuery,
    safeEvaluateQuery,
  }), []);

  return (
    <MotherDuckContext.Provider value={value}>
      {children}
    </MotherDuckContext.Provider>
  );
}

export function useMotherDuckClientState() {
  const context = useContext(MotherDuckContext);
  if (!context) {
    throw new Error('useMotherDuckClientState must be used within MotherDuckClientStateProvider');
  }
  return context;
} 