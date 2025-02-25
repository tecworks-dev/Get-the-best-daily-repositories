import { useTranslations } from 'next-intl';
import { useState, useEffect, useRef } from 'react';
import { MagnifyingGlassIcon, ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import { Modal } from './Modal';
import { OllamaService } from '@/types';

interface HeaderProps {
  countdown: number;
  detectingServices: Set<string>;
  detectedResults: OllamaService[];
  onDetect: (urls: string[]) => Promise<void>;
}

export function Header({ countdown, detectingServices, detectedResults, onDetect }: HeaderProps) {
  const t = useTranslations();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [detectResults, setDetectResults] = useState<OllamaService[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);

  // 记录已经更新过的服务器
  const updatedServersRef = useRef(new Set<string>());

  const handleDetect = async () => {
    const urls = urlInput.split('\n').filter(url => url.trim());
    if (urls.length === 0) return;
    
    setIsDetecting(true);
    try {
      // 过滤出新的服务地址
      const existingUrls = new Set(detectResults.map(result => result.server));
      const newUrls = urls.filter(url => !existingUrls.has(url));
      
      // 更新现有服务的状态为 loading
      setDetectResults(prev => prev.map(result => 
        urls.includes(result.server) 
          ? { ...result, loading: true, status: 'loading' as const }
          : result
      ));

      // 添加新的服务
      if (newUrls.length > 0) {
        const initialServices = newUrls.map(url => ({
          server: url,
          models: [],
          tps: 0,
          lastUpdate: new Date().toISOString(),
          loading: true,
          status: 'loading' as const
        }));
        setDetectResults(prev => [...prev, ...initialServices]);
      }
      
      // 开始检测
      await onDetect(urls);
    } finally {
      setIsDetecting(false);
    }
  };

  // 更新检测结果的状态
  useEffect(() => {
    setDetectResults(prev => 
      prev.map(result => {
        const isDetecting = detectingServices.has(result.server);
        // 查找最新的检测结果
        const latestResult = detectedResults.find(r => r.server === result.server);
        
        if (latestResult && !isDetecting) {
          // 如果检测成功且有可用模型，且状态发生了变化，且未更新过，异步更新服务器列表
          if (latestResult.status === 'success' && 
              latestResult.models.length > 0 && 
              result.status !== 'success' &&
              !updatedServersRef.current.has(latestResult.server)) {
            // 标记该服务器已更新
            updatedServersRef.current.add(latestResult.server);
            
            fetch('/api/update-servers', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ server: latestResult.server }),
            })
            .then(res => res.json())
            .then(data => {
              if (data.success) {
                console.log(data.exists 
                  ? `服务器已在监控列表中: ${latestResult.server}`
                  : `服务器已添加到监控列表: ${latestResult.server}`
                );
              }
            })
            .catch(error => {
              // 如果更新失败，移除标记以便下次重试
              updatedServersRef.current.delete(latestResult.server);
              console.error('更新服务器列表失败:', error);
            });
          }
          
          // 如果有最新结果且不在检测中，使用最新结果
          return {
            ...latestResult,
            loading: false,
            status: latestResult.models.length > 0 ? 'success' as const : 'error' as const
          };
        }
        
        return {
          ...result,
          loading: isDetecting,
          status: isDetecting ? 'loading' as const : result.models.length > 0 ? 'success' as const : 'error' as const
        };
      })
    );
  }, [detectingServices, detectedResults]);

  // 重置检测时清除已更新服务器的记录
  const handleNewDetection = () => {
    setDetectResults([]);
    setUrlInput('');
    updatedServersRef.current.clear();
  };

  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(detectResults, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ollama-detection-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  return (
    <div className="flex-1">
      <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-400">
        {t('title')}
      </h1>
      <button
        onClick={() => {
          setIsModalOpen(true);
          setDetectResults([]);
          setUrlInput('');
        }}
        disabled={isDetecting || countdown > 0}
        className={`mt-4 inline-flex items-center px-4 py-2 rounded-md shadow-sm text-sm font-medium
          transition-all duration-200 ease-in-out
          ${isDetecting || countdown > 0 
            ? 'bg-gray-600 cursor-not-allowed' 
            : 'bg-gradient-to-r from-blue-500 to-teal-500 hover:from-blue-600 hover:to-teal-600 text-white'
          }`}
      >
        <MagnifyingGlassIcon className={`-ml-1 mr-2 h-5 w-5 ${isDetecting ? 'animate-spin' : ''}`} />
        {isDetecting ? t('header.detecting') :
         countdown > 0 ? t('header.detectCountdown', { countdown }) : t('header.detect')}
      </button>

      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={t('detect.title')}
      >
        <div className="space-y-4">
          {detectResults.length === 0 ? (
            <>
              <p className="text-sm text-gray-400">
                {t('detect.description')}
              </p>
              <textarea
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder={t('detect.placeholder')}
                className="w-full h-40 px-3 py-2 text-sm bg-gray-700 border border-gray-600 rounded-md 
                  text-gray-200 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 
                  focus:border-transparent resize-none font-mono"
              />
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 text-sm font-medium text-gray-300 hover:text-white
                    transition-colors duration-200"
                >
                  {t('detect.cancel')}
                </button>
                <button
                  onClick={handleDetect}
                  disabled={!urlInput.trim() || isDetecting || countdown > 0}
                  className={`px-4 py-2 rounded-md text-sm font-medium
                    ${!urlInput.trim() || isDetecting || countdown > 0
                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-blue-500 to-teal-500 hover:from-blue-600 hover:to-teal-600 text-white'
                    }`}
                >
                  {isDetecting ? t('header.detecting') : 
                   countdown > 0 ? t('header.detectCountdown', { countdown }) : 
                   t('detect.confirm')}
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="max-h-[60vh] overflow-y-auto custom-scrollbar space-y-4">
                {detectResults.map((result) => (
                  <div key={result.server} className={`p-4 bg-gray-700 rounded-lg ${result.loading ? 'animate-pulse' : ''}`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <a
                          href={result.server}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-400 hover:text-blue-300 transition-colors duration-200"
                        >
                          {result.server}
                        </a>
                        {result.status === 'error' && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-500/20 text-red-400 border border-red-500/30">
                            {t('detect.error')}
                          </span>
                        )}
                        {result.status === 'success' && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-500/20 text-green-400 border border-green-500/30">
                            {t('detect.success')}
                          </span>
                        )}
                      </div>
                      <span className="text-sm text-gray-400">
                        {result.loading ? (
                          <div className="h-4 bg-gray-600 rounded animate-pulse w-16"></div>
                        ) : result.status === 'error' ? (
                          <span className="text-red-400">-</span>
                        ) : (
                          t('service.tps', { value: result.tps.toFixed(2) })
                        )}
                      </span>
                    </div>
                    <div className="mt-2">
                      <span className="text-sm text-gray-400">{t('service.availableModels')}</span>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {result.loading ? (
                          <div className="h-6 bg-gray-600 rounded animate-pulse w-24"></div>
                        ) : result.status === 'error' ? (
                          <span className="text-red-400">{t('detect.unavailable')}</span>
                        ) : (
                          result.models.map((model, idx) => (
                            <span
                              key={idx}
                              className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                                bg-gradient-to-r from-teal-500/20 to-blue-500/20 text-teal-400 border border-teal-500/30"
                            >
                              {model}
                            </span>
                          ))
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex justify-end space-x-3 pt-4 border-t border-gray-700">
                <button
                  onClick={handleNewDetection}
                  disabled={countdown > 0}
                  className={`px-4 py-2 text-sm font-medium
                    ${countdown > 0
                      ? 'text-gray-500 cursor-not-allowed'
                      : 'text-gray-300 hover:text-white transition-colors duration-200'
                    }`}
                >
                  {countdown > 0 
                    ? t('header.detectCountdown', { countdown })
                    : t('detect.newDetection')}
                </button>
                <button
                  onClick={handleDownload}
                  className="inline-flex items-center px-4 py-2 rounded-md text-sm font-medium
                    bg-gradient-to-r from-blue-500 to-teal-500 hover:from-blue-600 hover:to-teal-600 text-white"
                >
                  <ArrowDownTrayIcon className="h-5 w-5 mr-2" />
                  {t('detect.download')}
                </button>
              </div>
            </>
          )}
        </div>
      </Modal>
    </div>
  );
} 