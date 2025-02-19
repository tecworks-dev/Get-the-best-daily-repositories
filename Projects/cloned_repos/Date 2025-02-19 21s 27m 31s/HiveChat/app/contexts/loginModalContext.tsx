'use client';

import { createContext, useContext, useState } from 'react';

interface LoginModalContextType {
  visible: boolean;
  showLogin: () => void;
  hideLogin: () => void;
}

const LoginModalContext = createContext<LoginModalContextType>({
  visible: false,
  showLogin: () => {},
  hideLogin: () => {},
});

export function LoginModalProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [visible, setVisible] = useState(false);

  const showLogin = () => setVisible(true);
  const hideLogin = () => setVisible(false);

  return (
    <LoginModalContext.Provider value={{ visible, showLogin, hideLogin }}>
      {children}
    </LoginModalContext.Provider>
  );
}

export const useLoginModal = () => useContext(LoginModalContext);