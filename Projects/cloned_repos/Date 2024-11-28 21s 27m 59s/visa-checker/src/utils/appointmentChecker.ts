import type { VisaAppointment } from '../types';
import { config } from '../config/environment';
import { fetchAppointments } from '../services/api';
import { cacheService } from '../services/cache';
import { telegramService } from '../services/telegram';
import { extractCity } from './cityExtractor';

/**
 * Ana kontrol fonksiyonu
 * Yeni randevuları kontrol eder ve uygun olanları Telegram'a gönderir
 */
export async function checkAppointments(): Promise<void> {
  try {
    const appointments = await fetchAppointments();
    
    if (appointments.length === 0) {
      console.log('Randevu bulunamadı veya bir hata oluştu');
      return;
    }
    
    for (const appointment of appointments) {
      if (!isAppointmentValid(appointment)) continue;
      
      const appointmentKey = cacheService.createKey(appointment);
      
      // Debug modunda tüm randevuları göster
      if (config.app.debug) {
        console.log(`Randevu bulundu: ${JSON.stringify(appointment, null, 2)}`);
      }
      
      // Daha önce gönderilmemiş randevuları işle
      if (!cacheService.has(appointmentKey)) {
        await processNewAppointment(appointment, appointmentKey);
      }
    }
  } catch (error) {
    console.error('Randevu kontrolü sırasında hata:', error);
  }
}

/**
 * Randevunun geçerli olup olmadığını kontrol eder
 * @param appointment Kontrol edilecek randevu
 * @returns Randevu geçerli ise true, değilse false
 */
function isAppointmentValid(appointment: VisaAppointment): boolean {
  // Sadece hedef ülke için olan randevuları kontrol et
  if (appointment.source_country !== config.app.targetCountry) return false;

  // Sadece hedef misyon ülkesi için olan randevuları kontrol et
  if (appointment.mission_country !== config.app.missionCountry) return false;

  // Eğer hedef şehirler belirtilmişse, sadece o şehirlerdeki randevuları kontrol et
  if (config.app.targetCities.length > 0) {
    const appointmentCity = extractCity(appointment.center_name);
    const cityMatch = config.app.targetCities.some(city => 
      appointmentCity.toLowerCase().includes(city.toLowerCase())
    );
    if (!cityMatch) return false;
  }

  return true;
}

/**
 * Yeni randevuyu işler ve Telegram'a gönderir
 * @param appointment İşlenecek randevu
 * @param appointmentKey Randevu için önbellek anahtarı
 */
async function processNewAppointment(appointment: VisaAppointment, appointmentKey: string): Promise<void> {
  cacheService.set(appointmentKey);
  
  const success = await telegramService.sendNotification(appointment);
  if (success) {
    console.log(`Bildirim gönderildi: ${appointmentKey}`);
  } else {
    // Hata durumunda önbellekten sil ve bir sonraki kontrolde tekrar dene
    cacheService.delete(appointmentKey);
  }
} 