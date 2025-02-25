import { useTranslations } from 'next-intl';
import { formatDistanceToNow } from 'date-fns';
import { zhCN, enUS } from 'date-fns/locale';
import { PAGE_SIZE_OPTIONS } from '@/constants';
import { useParams } from 'next/navigation';
import { OllamaService } from '@/types';

interface ServiceListProps {
  services: OllamaService[];
  currentPage: number;
  pageSize: number;
  totalPages: number;
  isClient: boolean;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
}

export function ServiceList({
  services,
  currentPage,
  pageSize,
  totalPages,
  isClient,
  onPageChange,
  onPageSizeChange,
}: ServiceListProps) {
  const t = useTranslations();
  const params = useParams();
  const locale = params.locale as string;

  // const formatDate = (date: string) => {
  //   if (!isClient) {
  //     return new Date(date).toISOString();
  //   }
  //   return formatDistanceToNow(new Date(date), {
  //     addSuffix: true,
  //     locale: locale === 'zh' ? zhCN : enUS,
  //   });
  // };

  // 生成页码数组
  const getPageNumbers = () => {
    const pages = [];
    const maxVisiblePages = 5;
    
    if (totalPages <= maxVisiblePages) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      if (currentPage <= 3) {
        for (let i = 1; i <= 4; i++) pages.push(i);
        pages.push('...');
        pages.push(totalPages);
      } else if (currentPage >= totalPages - 2) {
        pages.push(1);
        pages.push('...');
        for (let i = totalPages - 3; i <= totalPages; i++) pages.push(i);
      } else {
        pages.push(1);
        pages.push('...');
        for (let i = currentPage - 1; i <= currentPage + 1; i++) pages.push(i);
        pages.push('...');
        pages.push(totalPages);
      }
    }
    return pages;
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-700">
        <thead className="bg-gray-800">
          <tr>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
              {t('service.server')}
            </th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
              {t('service.models')}
            </th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
              TPS
            </th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
              {t('service.lastUpdate', { value: '' })}
            </th>
          </tr>
        </thead>
        <tbody className="bg-gray-900 divide-y divide-gray-800">
          {services.map((service, index) => (
            <tr key={service.server} className={`${index % 2 === 0 ? 'bg-gray-900' : 'bg-gray-850'} 
              ${service.loading ? 'animate-pulse' : ''}`}>
              <td className="px-6 py-4 whitespace-nowrap text-sm">
                <a
                  href={service.server}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 transition-colors duration-200"
                >
                  {service.server}
                </a>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                {service.loading ? (
                  <div className="h-6 bg-gray-700 rounded animate-pulse w-24"></div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {service.models.map((model, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                          bg-gradient-to-r from-teal-500/20 to-blue-500/20 text-teal-400 border border-teal-500/30"
                      >
                        {model}
                      </span>
                    ))}
                  </div>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                {service.loading ? (
                  <div className="h-4 bg-gray-700 rounded animate-pulse w-16"></div>
                ) : (
                  t('service.tps', { value: service.tps.toFixed(2) })
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                {service.loading ? (
                  <div className="h-4 bg-gray-700 rounded animate-pulse w-32"></div>
                ) : (
                  isClient && (
                    <time dateTime={service.lastUpdate}>
                      {t('service.lastUpdateValue', {
                        value: formatDistanceToNow(new Date(service.lastUpdate), {
                          addSuffix: true,
                          locale: locale === 'zh' ? zhCN : enUS,
                        })
                      })}
                    </time>
                  )
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* 分页控制 */}
      <div className="p-4 border-b border-gray-700 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <span className="text-gray-400">{t('pagination.perPage')}</span>
          <select
            value={pageSize}
            onChange={(e) => onPageSizeChange(Number(e.target.value))}
            className="bg-gray-700 border border-gray-600 rounded-md text-gray-200 px-2 py-1
              focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {PAGE_SIZE_OPTIONS.map(size => (
              <option key={size} value={size}>{size}</option>
            ))}
          </select>
        </div>
        <div className="text-gray-400">
          {t('pagination.showing', {
            from: (currentPage - 1) * pageSize + 1,
            to: Math.min(currentPage * pageSize, services.length),
            total: services.length
          })}
        </div>
      </div>

      {/* 分页导航 */}
      {totalPages > 1 && (
        <div className="p-4 border-t border-gray-700 flex justify-center">
          <div className="flex space-x-2">
            <button
              onClick={() => onPageChange(1)}
              disabled={currentPage === 1}
              className={`px-3 py-1 rounded-md text-sm font-medium
                ${currentPage === 1
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
            >
              {t('pagination.first')}
            </button>
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage === 1}
              className={`px-3 py-1 rounded-md text-sm font-medium
                ${currentPage === 1
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
            >
              {t('pagination.prev')}
            </button>
            {getPageNumbers().map((page, index) => (
              <button
                key={index}
                onClick={() => typeof page === 'number' && onPageChange(page)}
                disabled={page === '...'}
                className={`px-3 py-1 rounded-md text-sm font-medium
                  ${page === currentPage
                    ? 'bg-gradient-to-r from-blue-500 to-teal-500 text-white'
                    : page === '...'
                      ? 'bg-gray-700 text-gray-500 cursor-default'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
              >
                {page}
              </button>
            ))}
            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
              className={`px-3 py-1 rounded-md text-sm font-medium
                ${currentPage === totalPages
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
            >
              {t('pagination.next')}
            </button>
            <button
              onClick={() => onPageChange(totalPages)}
              disabled={currentPage === totalPages}
              className={`px-3 py-1 rounded-md text-sm font-medium
                ${currentPage === totalPages
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
            >
              {t('pagination.last')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
} 