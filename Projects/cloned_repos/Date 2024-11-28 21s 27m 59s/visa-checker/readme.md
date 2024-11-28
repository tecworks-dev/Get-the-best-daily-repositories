# 🔍 Schengen Vize Randevu Takip Botu

Bu bot, Schengen vizesi için randevu durumlarını otomatik olarak takip eder ve yeni randevular açıldığında Telegram üzerinden bildirim gönderir.

## 📋 Özellikler

- 🔄 Otomatik randevu kontrolü
- 🌍 Çoklu şehir desteği
- 🇪🇺 Farklı Schengen ülkeleri için randevu takibi
- 📱 Telegram üzerinden anlık bildirimler
- ⏰ Özelleştirilebilir kontrol sıklığı
- 🚫 Rate limit koruması
- 🔍 Detaylı hata ayıklama modu

## 🛠 Sistem Gereksinimleri

### Yazılım Gereksinimleri
- Node.js (v16 veya üzeri)
- Paket yöneticisi (npm, yarn veya pnpm)
- Telegram Bot Token'ı
- Telegram Kanal/Grup ID'si

### Donanım/Hosting Gereksinimleri
Bot'un sürekli çalışabilmesi için aşağıdaki seçeneklerden birine ihtiyacınız var:

1. **VPS (Virtual Private Server) - Önerilen 🌟**
   - 7/24 kesintisiz çalışma
   - Düşük maliyetli (aylık 50-100 lira)
   - Önerilen sağlayıcılar (dolar bazlı): DigitalOcean, Linode, Vultr, OVH
   - Önerilen sağlayıcılar (türk lirası bazlı): DeHost, Natro, Turhost

2. **Kişisel Bilgisayar**
   - 7/24 açık kalması gerekir
   - Elektrik kesintilerinden etkilenir
   - İnternet bağlantısı sürekli olmalı
   - Bilgisayarın uyku moduna geçmesi engellenmelidir

3. **Raspberry Pi**
   - Düşük güç tüketimi
   - 7/24 çalıştırılabilir
   - Ekonomik çözüm
   - Kurulum biraz teknik bilgi gerektirir

> ⚠️ **Önemli Not**: Bot'un randevuları kaçırmaması için sürekli çalışır durumda olması gerekir. VPS kullanımı, kesintisiz çalışma ve düşük maliyet açısından en ideal çözümdür.

## 🛠️ Kurulum

### Gereksinimler

- Node.js (v16 veya üzeri)
- Paket yöneticisi (npm, yarn veya pnpm)
- Telegram Bot Token'ı
- Telegram Kanal/Grup ID'si

### 1. Telegram Bot Oluşturma

1. Telegram'da [@BotFather](https://t.me/botfather) ile konuşma başlatın
2. `/newbot` komutunu gönderin
3. Bot için bir isim belirleyin
4. Bot için bir kullanıcı adı belirleyin (sonu 'bot' ile bitmeli)
5. BotFather size bir token verecek, bu token'ı kaydedin

### 2. Telegram Kanal ID'si Alma

1. Bir Telegram kanalı oluşturun
2. Botu kanala ekleyin ve admin yapın
3. Kanala bir mesaj gönderin
4. Bu URL'yi ziyaret edin: `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`
   - `<BOT_TOKEN>` yerine botunuzun token'ını yazın
5. JSON çıktısında `"chat":{"id":-100xxxxxxxxxx}` şeklinde bir değer göreceksiniz
6. Bu ID'yi kaydedin (örn: -100xxxxxxxxxx)

### 3. Projeyi Kurma

1. Projeyi bilgisayarınıza indirin:
```bash
git clone https://github.com/byigitt/visa-checker.git
cd visa-checker
```

2. Gerekli paketleri yükleyin:
```bash
# npm kullanıyorsanız
npm install

# yarn kullanıyorsanız
yarn install

# pnpm kullanıyorsanız
pnpm install
```

3. `.env.example` dosyasını `.env` olarak kopyalayın:
```bash
cp .env.example .env
```

4. `.env` dosyasını düzenleyin:
```env
# Telegram Yapılandırması
TELEGRAM_BOT_TOKEN=your_bot_token_here        # Telegram bot token'ınız
TELEGRAM_CHAT_ID=your_chat_id_here            # Telegram kanal ID'niz (örn: -100123456789)
TELEGRAM_RATE_LIMIT=20                        # Telegram API için dakikada maksimum mesaj sayısı
TELEGRAM_RETRY_AFTER=5000                     # Rate limit aşımında beklenecek süre (milisaniye)
TELEGRAM_RATE_LIMIT_MINUTES=15                # Bildirimler arası minimum süre (dakika)

# Uygulama Yapılandırması
CHECK_INTERVAL=*/5 * * * *                    # Kontrol sıklığı (varsayılan: her 5 dakikada bir)
TARGET_COUNTRY=Turkiye                        # Kaynak ülke (değiştirmeyin)

# Randevu Filtreleme
CITIES=Ankara,Istanbul                        # Takip edilecek şehirler (virgülle ayırın)
MISSION_COUNTRY=Netherlands                   # Randevusu takip edilecek ülke

# API Yapılandırması
VISA_API_URL=https://api.schengenvisaappointments.com/api/visa-list/?format=json

# Önbellek Yapılandırması
MAX_CACHE_SIZE=1000                          # Maksimum önbellek boyutu
CACHE_CLEANUP_INTERVAL=86400000              # Önbellek temizleme sıklığı (ms)
MAX_RETRIES=3                                # API hatası durumunda maksimum deneme sayısı
RETRY_DELAY_BASE=1000                        # API hatası durumunda bekleme süresi (ms)

# Hata Ayıklama
DEBUG=false                                  # Hata ayıklama modu (true/false)
```

5. TypeScript kodunu derleyin:
```bash
# npm kullanıyorsanız
npm run build

# yarn kullanıyorsanız
yarn build

# pnpm kullanıyorsanız
pnpm build
```

### 4. Botu Çalıştırma

1. Geliştirme modunda çalıştırma:
```bash
# npm kullanıyorsanız
npm run dev

# yarn kullanıyorsanız
yarn dev

# pnpm kullanıyorsanız
pnpm dev
```

2. Production modunda çalıştırma:
```bash
# npm kullanıyorsanız
npm start

# yarn kullanıyorsanız
yarn start

# pnpm kullanıyorsanız
pnpm start
```

## ⚙️ Yapılandırma Seçenekleri

### Telegram Ayarları
- `TELEGRAM_BOT_TOKEN`: Telegram bot token'ınız
- `TELEGRAM_CHAT_ID`: Telegram kanal ID'niz
- `TELEGRAM_RATE_LIMIT`: Dakikada gönderilebilecek maksimum mesaj sayısı
- `TELEGRAM_RETRY_AFTER`: Rate limit aşıldığında beklenecek süre (ms)
- `TELEGRAM_RATE_LIMIT_MINUTES`: Bildirimler arası minimum süre

### Randevu Takip Ayarları
- `CHECK_INTERVAL`: Randevu kontrolü sıklığı (cron formatında)
- `CITIES`: Takip edilecek şehirler (virgülle ayrılmış liste)
- `MISSION_COUNTRY`: Randevusu takip edilecek ülke

### Sistem Ayarları
- `MAX_CACHE_SIZE`: Önbellekteki maksimum randevu sayısı
- `CACHE_CLEANUP_INTERVAL`: Önbellek temizleme sıklığı (ms)
- `MAX_RETRIES`: API hatalarında tekrar deneme sayısı
- `RETRY_DELAY_BASE`: API hataları arasında bekleme süresi
- `DEBUG`: Detaylı log kayıtları için hata ayıklama modu

## 📱 Bildirim Örneği

Bot, yeni bir randevu bulduğunda şu formatta bir mesaj gönderir:

```
🔔 Yeni Randevu Bildirimi

📍 Merkez: Netherlands Visa Application Centre - Ankara
🗓 Tarih: 24 Aralık 2024
🎫 Kategori: KISA DONEM VIZE / SHORT TERM VISA
📋 Alt Kategori: TURIZM VIZE BASVURUSU / TOURISM VISA APPLICATION
👥 Bekleyen Kişi: 5

🔗 Randevu Linki:
https://visa.vfsglobal.com/tur/en/nld/login
```

## 🤔 Sık Sorulan Sorular

1. **Bot çalışıyor mu?**
   - Konsolda "Vize randevu kontrolü başlatıldı" mesajını görmelisiniz
   - Debug modunu aktif ederek daha detaylı loglar görebilirsiniz

2. **Telegram bildirimleri gelmiyor**
   - Bot token'ınızı kontrol edin
   - Kanal ID'sini kontrol edin
   - Botun kanalda admin olduğundan emin olun

3. **Belirli bir şehir/ülke için randevuları nasıl takip ederim?**
   - `.env` dosyasında `CITIES` ve `MISSION_COUNTRY` değerlerini düzenleyin

4. **Rate limit hatası alıyorum**
   - `TELEGRAM_RATE_LIMIT_MINUTES` değerini artırın
   - Kontrol sıklığını azaltın

## 🚨 Hata Bildirimi

Bir hata bulduysanız veya öneriniz varsa, lütfen GitHub üzerinden issue açın.

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Daha fazla bilgi için [LICENSE](LICENSE) dosyasına bakın.
