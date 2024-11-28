/**
 * Randevu bilgilerini içeren tip tanımı
 */
export interface VisaAppointment {
  source_country: string;      // Kaynak ülke (örn: Turkiye)
  mission_country: string;     // Hedef ülke (örn: Netherlands)
  center_name: string;         // Merkez adı
  appointment_date: string;    // Randevu tarihi
  visa_type_id: number;       // Vize tipi ID'si
  visa_category: string;      // Vize kategorisi
  visa_subcategory: string;   // Vize alt kategorisi
  people_looking: number;     // Bekleyen kişi sayısı
  book_now_link: string;      // Randevu alma linki
  last_checked: string;       // Son kontrol tarihi
} 

export interface AppointmentCache {
  [key: string]: boolean;
} 