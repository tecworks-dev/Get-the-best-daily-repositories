import React, { createContext, useState } from "react";

interface ViewContextType {
  activeView: View;
  setActiveView: React.Dispatch<React.SetStateAction<View>>;
}
const ViewContext = createContext<ViewContextType | undefined>(undefined);

const ViewProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [activeView, setActiveView] = useState<View>("Chat");

  return (
    <ViewContext.Provider value={{ activeView, setActiveView }}>
      {children}
    </ViewContext.Provider>
  );
};

export { ViewProvider, ViewContext };
