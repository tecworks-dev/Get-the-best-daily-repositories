import type { AppointmentCache } from '../types';
import { config } from '../config/environment';

/**
 * Önbellek Servisi
 * Daha önce gönderilen randevuları takip eder ve tekrar gönderilmesini engeller
 */
class CacheService {
  private cache: AppointmentCache = {};

  /**
   * Randevu bilgilerinden benzersiz bir anahtar oluşturur
   * Bu anahtar randevunun daha önce gönderilip gönderilmediğini kontrol etmek için kullanılır
   */
  createKey(params: { 
    source_country: string; 
    mission_country: string; 
    center_name: string; 
    appointment_date: string; 
  }): string {
    return `${params.source_country}-${params.mission_country}-${params.center_name}-${params.appointment_date}`;
  }

  /**
   * Belirtilen anahtarın önbellekte olup olmadığını kontrol eder
   */
  has(key: string): boolean {
    return !!this.cache[key];
  }

  /**
   * Yeni bir randevuyu önbelleğe ekler
   */
  set(key: string): void {
    this.cache[key] = true;
  }

  /**
   * Belirtilen anahtarı önbellekten siler
   */
  delete(key: string): void {
    delete this.cache[key];
  }

  /**
   * Önbelleği temizler:
   * 1. Dünden önceki randevuları siler
   * 2. Maksimum önbellek boyutunu aşan durumlarda en eski kayıtları siler
   */
  cleanup(): void {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    
    const cacheEntries = Object.entries(this.cache);
    
    // Eski kayıtları sil
    for (const [key, _] of cacheEntries) {
      const appointmentDate = key.split('-')[3];
      if (new Date(appointmentDate) < yesterday) {
        this.delete(key);
      }
    }
    
    // Önbellek boyutu aşıldıysa en eski kayıtları sil
    if (Object.keys(this.cache).length > config.cache.maxSize) {
      const sortedKeys = Object.keys(this.cache)
        .sort((a, b) => new Date(a.split('-')[3]).getTime() - new Date(b.split('-')[3]).getTime());
      
      while (Object.keys(this.cache).length > config.cache.maxSize) {
        const oldestKey = sortedKeys.shift();
        if (oldestKey) this.delete(oldestKey);
      }
    }
  }

  /**
   * Düzenli temizleme işlemini başlatır
   * Belirlenen aralıklarla önbelleği temizler
   */
  startCleanupInterval(): void {
    setInterval(() => this.cleanup(), config.cache.cleanupInterval);
  }
}

export const cacheService = new CacheService(); 