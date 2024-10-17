import { AVAILABLE_LOCALES } from "@/framework/locale/locale";

export interface PaymentRef {
  open: () => void,
  locale: AVAILABLE_LOCALES
}