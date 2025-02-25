'use client';

import { useState, useEffect } from 'react';
import { useTranslations } from 'next-intl';
import { Header } from '@/components/Header';
import { ModelFilter } from '@/components/ModelFilter';
import { ServiceList } from '@/components/ServiceList';
import { Footer } from '@/components/Footer';
import { LanguageSwitch } from '@/components/LanguageSwitch';
import { OllamaService, SortField, SortOrder } from '@/types';
import { useParams } from 'next/navigation';


export default function Home() {
  const t = useTranslations();
  const [services, setServices] = useState<OllamaService[]>([]);
  const [countdown, setCountdown] = useState(0);
  const [detectingServices, setDetectingServices] = useState<Set<string>>(new Set());
  const [detectedResults, setDetectedResults] = useState<OllamaService[]>([]);

  const { locale } = useParams();
  
  // 排序状态
  const [sortField, setSortField] = useState<SortField>('tps');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  
  // 过滤状态
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  // 分页状态
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  // 客户端渲染标记
  const [isClient, setIsClient] = useState(false);

  const fetchData = async () => {
    try {
      // 使用绝对路径
      const response = await fetch('/data.json', {
        // 添加 cache 控制
        cache: 'no-store', // 或者使用 'force-cache' 如果你想要缓存
      });
      const data = await response.json();
      // 确保所有服务都有 loading 属性
      const servicesWithLoading = data.map((service: OllamaService) => ({
        ...service,
        loading: false,
        status: service.models.length > 0 ? 'success' : 'error'
      }));
      setServices(servicesWithLoading);
      
      // 更新可用模型列表
      const models = new Set<string>();
      servicesWithLoading.forEach((service: OllamaService) => {
        service.models.forEach(model => models.add(model));
      });
      setAvailableModels(Array.from(models).sort());
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleDetect = async (urls: string[]): Promise<void> => {
    try {
      setDetectingServices(new Set(urls));
      
      for (const url of urls) {
        try {
          const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url }),
          });

          if (!response.ok) {
            console.error(`检测失败: ${url}, 状态码: ${response.status}`);
            setDetectedResults(prev => [...prev, {
              server: url,
              models: [],
              tps: 0,
              lastUpdate: new Date().toISOString(),
              status: 'error'
            }]);
            continue;
          }

          const result = await response.json();
          setDetectedResults(prev => [...prev, result]);

        } catch (error) {
          console.error(`检测出错: ${url}`, error);
          setDetectedResults(prev => [...prev, {
            server: url,
            models: [],
            tps: 0,
            lastUpdate: new Date().toISOString(),
            status: 'error'
          }]);
        } finally {
          setDetectingServices(prev => {
            const next = new Set(prev);
            next.delete(url);
            return next;
          });
        }
      }
      
      setCountdown(5);
    } catch (error) {
      console.error('检测过程出错:', error);
      setDetectingServices(new Set());
    }
  };

  useEffect(() => {
    // 只在客户端执行
    if (typeof window !== 'undefined') {
      fetchData();
    }
  }, []);

  useEffect(() => {
    if (countdown > 0) {
      const timer = setInterval(() => {
        setCountdown(prev => prev - 1);
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [countdown]);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // 排序函数
  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  // 切换模型选择
  const toggleModelSelection = (model: string) => {
    setSelectedModels(prev => 
      prev.includes(model)
        ? prev.filter(m => m !== model)
        : [...prev, model]
    );
  };

  // 移除单个选中的模型
  const removeSelectedModel = (model: string) => {
    setSelectedModels(prev => prev.filter(m => m !== model));
  };

  // 清空所有选中的模型
  const clearSelectedModels = () => {
    setSelectedModels([]);
  };

  // 过滤和排序服务列表
  const filteredAndSortedServices = services
    .filter(service => 
      (selectedModels.length === 0 || 
      service.models.some(model => selectedModels.includes(model)))
    )
    .sort((a, b) => {
      const multiplier = sortOrder === 'asc' ? 1 : -1;
      // 将 loading 状态的服务排在最前面
      if (a.loading && !b.loading) return -1;
      if (!a.loading && b.loading) return 1;
      if (sortField === 'tps') {
        return (a.tps - b.tps) * multiplier;
      } else {
        return (new Date(a.lastUpdate).getTime() - new Date(b.lastUpdate).getTime()) * multiplier;
      }
    });

  // 计算分页数据
  const paginatedServices = filteredAndSortedServices.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  // 计算总页数
  const totalPages = Math.ceil(filteredAndSortedServices.length / pageSize);

  // 页面改变处理
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // 每页显示数量改变处理
  const handlePageSizeChange = (newSize: number) => {
    setPageSize(newSize);
    setCurrentPage(1);
  };

  // 当过滤条件改变时，重置页码
  useEffect(() => {
    setCurrentPage(1);
  }, [selectedModels, sortField, sortOrder]);

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center mb-8">
          <Header
            countdown={countdown}
            detectingServices={detectingServices}
            detectedResults={detectedResults}
            onDetect={handleDetect}
          />
          <LanguageSwitch currentLocale={locale as "en" | "zh"} />
        </div>

        <div className="bg-gray-800 shadow-xl rounded-xl overflow-hidden border border-gray-700 mb-6">
          <ModelFilter
            selectedModels={selectedModels}
            availableModels={availableModels}
            searchTerm={searchTerm}
            sortField={sortField}
            sortOrder={sortOrder}
            onSearchChange={setSearchTerm}
            onToggleModel={toggleModelSelection}
            onRemoveModel={removeSelectedModel}
            onClearModels={clearSelectedModels}
            onToggleSort={toggleSort}
          />

          <ServiceList
            services={paginatedServices}
            currentPage={currentPage}
            pageSize={pageSize}
            totalPages={totalPages}
            isClient={isClient}
            onPageChange={handlePageChange}
            onPageSizeChange={handlePageSizeChange}
          />
        </div>
        
        <div className="text-sm text-gray-400 text-right">
          {t('service.total', { count: filteredAndSortedServices.length })}
          {selectedModels.length > 0 && t('service.filtered', { models: selectedModels.join(', ') })}
        </div>

        <Footer />
      </div>
    </main>
  );
} 