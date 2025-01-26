"use client";

import { createContext, useEffect, useState } from "react";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isLoading, setLoading] = useState(true);

  useEffect(() => {
    const checkLoginStatus = async () => {
      const savedLoginStatus = localStorage.getItem("isLoggedIn");
      setIsLoggedIn(savedLoginStatus === "true");
      setLoading(false);
    };

    checkLoginStatus();
  }, []);

  const login = () => {
    setIsLoggedIn(true);
    //TODO: implement login logic
    localStorage.setItem("isLoggedIn", "true");
  };

  const logout = () => {
    setIsLoggedIn(false);
    localStorage.setItem("isLoggedIn", "false");
  };

  return (
    <AuthContext.Provider value={{ isLoggedIn, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
