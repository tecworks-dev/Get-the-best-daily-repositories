import { useTranslations } from 'next-intl';

export function Footer() {
  const t = useTranslations();
  const disclaimerItems = t.raw('footer.disclaimer.items') as string[];

  return (
    <footer className="mt-12 border-t border-gray-700 pt-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* 作者信息 */}
          <div>
            <h3 className="text-lg font-semibold text-gray-200 mb-4">{t('footer.about.title')}</h3>
            <div className="space-y-2">
              <p className="text-gray-400">
                {t('footer.about.author')}
                <a 
                  href="https://github.com/forrany"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 transition-colors duration-200"
                >
                  VincentKo (@forrany)
                </a>
              </p>
              <p className="text-gray-400">
                {t('footer.about.license')}
              </p>
            </div>
          </div>

          {/* 免责声明 */}
          <div>
            <h3 className="text-lg font-semibold text-gray-200 mb-4">{t('footer.disclaimer.title')}</h3>
            <div className="space-y-2 text-gray-400 text-sm">
              {disclaimerItems.map((item: string, index: number) => (
                <p key={index}>{item}</p>
              ))}
            </div>
          </div>
        </div>

        {/* 版权信息 */}
        <div className="mt-8 pt-8 border-t border-gray-700">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <p className="text-gray-400 text-sm">
              {t('footer.copyright', { year: new Date().getFullYear() })}
            </p>
            <div className="flex items-center space-x-4">
              <a
                href="https://github.com/forrany"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-gray-300 transition-colors duration-200"
              >
                GitHub
              </a>
              <span className="text-gray-600">|</span>
              <a
                href="https://github.com/forrany"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-gray-300 transition-colors duration-200"
              >
                MIT License
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
} 