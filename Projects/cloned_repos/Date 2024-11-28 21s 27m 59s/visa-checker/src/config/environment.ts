import dotenv from 'dotenv';

dotenv.config();

/**
 * Çevre değişkenleri için tip tanımlamaları
 */
export interface EnvironmentConfig {
  // Telegram ile ilgili yapılandırmalar
  telegram: {
    botToken: string;      // Telegram bot token'ı
    channelId: string;     // Telegram kanal ID'si
    rateLimit: number;     // Dakikada gönderilebilecek maksimum mesaj sayısı
    retryAfter: number;    // Rate limit aşıldığında beklenecek süre (ms)
  };
  // Uygulama genel yapılandırmaları
  app: {
    checkInterval: string;    // Kontrol sıklığı (cron formatında)
    targetCountry: string;    // Kaynak ülke (Turkiye)
    targetCities: string[];   // Takip edilecek şehirler listesi
    missionCountry: string;   // Hedef ülke (Netherlands, France vb.)
    debug: boolean;           // Hata ayıklama modu
  };
  // API ile ilgili yapılandırmalar
  api: {
    visaApiUrl: string;      // Vize API'sinin adresi
    maxRetries: number;      // Maksimum deneme sayısı
    retryDelayBase: number;  // Denemeler arası bekleme süresi (ms)
  };
  // Önbellek yapılandırmaları
  cache: {
    maxSize: number;         // Maksimum önbellek boyutu
    cleanupInterval: number; // Temizleme sıklığı (ms)
  };
}

/**
 * Çevre değişkenlerini doğrular ve yapılandırma nesnesini oluşturur
 * @returns Doğrulanmış yapılandırma nesnesi
 * @throws Eksik veya hatalı yapılandırma durumunda hata fırlatır
 */
function validateEnvironment(): EnvironmentConfig {
  // Zorunlu çevre değişkenlerini kontrol et
  const requiredEnvVars = {
    TELEGRAM_BOT_TOKEN: process.env.TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID: process.env.TELEGRAM_CHAT_ID,
  };

  // Eksik değişkenleri bul
  const missingVars = Object.entries(requiredEnvVars)
    .filter(([_, value]) => !value)
    .map(([key]) => key);

  // Eksik değişken varsa hata fırlat
  if (missingVars.length > 0) {
    console.error(`Eksik çevre değişkenleri: ${missingVars.join(', ')}`);
    process.exit(1);
  }

  // Telegram kanal ID'sini doğrula
  const channelId = process.env.TELEGRAM_CHAT_ID;
  if (!channelId || !/^-?\d+$/.test(channelId)) {
    console.error('Geçersiz TELEGRAM_CHAT_ID formatı');
    process.exit(1);
  }

  // Şehirleri virgülle ayrılmış listeden diziye çevir
  const cities = process.env.CITIES ? process.env.CITIES.split(',').map(city => city.trim()) : [];

  // Yapılandırma nesnesini oluştur ve döndür
  return {
    telegram: {
      botToken: process.env.TELEGRAM_BOT_TOKEN as string,
      channelId,
      rateLimit: Number(process.env.TELEGRAM_RATE_LIMIT_MINUTES) || 15,
      retryAfter: Number(process.env.TELEGRAM_RETRY_AFTER) || 5000,
    },
    app: {
      checkInterval: process.env.CHECK_INTERVAL || '*/5 * * * *',
      targetCountry: process.env.TARGET_COUNTRY || 'Turkiye',
      targetCities: cities,
      missionCountry: process.env.MISSION_COUNTRY || 'Netherlands',
      debug: process.env.DEBUG === 'true',
    },
    api: {
      visaApiUrl: process.env.VISA_API_URL || 'https://api.schengenvisaappointments.com/api/visa-list/?format=json',
      maxRetries: Number(process.env.MAX_RETRIES) || 3,
      retryDelayBase: Number(process.env.RETRY_DELAY_BASE) || 1000,
    },
    cache: {
      maxSize: Number(process.env.MAX_CACHE_SIZE) || 1000,
      cleanupInterval: Number(process.env.CACHE_CLEANUP_INTERVAL) || 24 * 60 * 60 * 1000,
    },
  };
}

// Yapılandırma nesnesini oluştur ve dışa aktar
export const config = validateEnvironment(); 