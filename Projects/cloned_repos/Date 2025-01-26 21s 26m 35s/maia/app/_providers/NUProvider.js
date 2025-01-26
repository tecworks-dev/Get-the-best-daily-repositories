import { NextUIProvider } from "@nextui-org/react";

export default function NUProvider({ children }) {
  return <NextUIProvider>{children}</NextUIProvider>;
}
