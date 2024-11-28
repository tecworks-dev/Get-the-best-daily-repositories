import axios, { AxiosError } from 'axios';
import type { VisaAppointment } from '../types';
import { config } from '../config/environment';

/**
 * API isteklerini yeniden deneme mekanizması
 * Sunucu hatası durumunda belirli sayıda tekrar dener
 * @param fn API çağrısını yapan fonksiyon
 * @param retries Kalan deneme sayısı
 */
async function fetchWithRetry<T>(
  fn: () => Promise<T>,
  retries = config.api.maxRetries
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (retries > 0 && 
        error instanceof AxiosError && 
        error.response && 
        typeof error.response.status === 'number' && 
        error.response.status >= 500) {
      console.log(`Yeniden deneniyor... ${config.api.maxRetries - retries + 1}/${config.api.maxRetries}`);
      await new Promise(resolve => 
        setTimeout(resolve, config.api.retryDelayBase * (config.api.maxRetries - retries + 1))
      );
      return fetchWithRetry(fn, retries - 1);
    }
    throw error;
  }
}

/**
 * Vize randevularını API'den çeker
 * Hata durumunda boş dizi döner ve hatayı loglar
 */
export async function fetchAppointments(): Promise<VisaAppointment[]> {
  try {
    const response = await fetchWithRetry(() => 
      axios.get<VisaAppointment[]>(config.api.visaApiUrl)
    );
    return response.data;
  } catch (error) {
    if (error instanceof AxiosError) {
      console.error('API Hatası:', {
        durum: error.response?.status,
        mesaj: error.message,
        url: error.config?.url
      });
    } else {
      console.error('Bilinmeyen hata:', error);
    }
    return [];
  }
} 