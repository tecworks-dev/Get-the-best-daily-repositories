'use client'
import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { User, UserType } from '@prisma/client';

interface AuthUser {
  id: string;
  name: string | null;
  email: string | null;
  credit: number;
  type: UserType;
}

interface AuthContextType {
  user: AuthUser | null;
  loading: boolean;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchUser = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/user');
      if (response.ok) {
        const userData: User = await response.json();
        setUser({
          id: userData.id,
          name: userData.name,
          email: userData.email,
          credit: userData.credit,
          type: userData.type,
        });
      } else {
        setUser(null);
      }
    } catch (error) {
      console.error('获取用户信息失败:', error);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  const refreshUser = useCallback(() => {
    return fetchUser();
  }, [fetchUser]);

  return (
    <AuthContext.Provider value={{ user, loading, refreshUser }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuthUser = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuthUser must be used within an AuthProvider');
  }
  return context;
};

export default useAuthUser;