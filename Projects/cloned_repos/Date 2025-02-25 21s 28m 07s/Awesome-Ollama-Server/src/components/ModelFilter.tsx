import { useTranslations } from 'next-intl';
import { ArrowUpIcon, ArrowDownIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface ModelFilterProps {
  selectedModels: string[];
  availableModels: string[];
  searchTerm: string;
  sortField: 'tps' | 'lastUpdate';
  sortOrder: 'asc' | 'desc';
  onSearchChange: (term: string) => void;
  onToggleModel: (model: string) => void;
  onRemoveModel: (model: string) => void;
  onClearModels: () => void;
  onToggleSort: (field: 'tps' | 'lastUpdate') => void;
}

export function ModelFilter({
  selectedModels,
  availableModels,
  searchTerm,
  sortField,
  sortOrder,
  onSearchChange,
  onToggleModel,
  onRemoveModel,
  onClearModels,
  onToggleSort,
}: ModelFilterProps) {
  const t = useTranslations();

  // 获取过滤后的模型列表
  const filteredModels = availableModels.filter(model => 
    model.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // 排序图标组件
  const SortIcon = ({ field }: { field: 'tps' | 'lastUpdate' }) => {
    if (sortField !== field) return null;
    return sortOrder === 'asc' ? 
      <ArrowUpIcon className="h-4 w-4 inline-block ml-1" /> :
      <ArrowDownIcon className="h-4 w-4 inline-block ml-1" />;
  };

  return (
    <div className="p-6 border-b border-gray-700 space-y-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium text-gray-200">{t('filter.title')}</h2>
          <div className="flex space-x-4">
            <button
              onClick={() => onToggleSort('tps')}
              className={`px-4 py-2 text-sm font-medium rounded-md border border-gray-600
                transition-all duration-200 hover:bg-gray-700
                ${sortField === 'tps' ? 'bg-gray-700 text-blue-400' : 'text-gray-300'}`}
            >
              {t('filter.sort.tps')}
              <SortIcon field="tps" />
            </button>
            <button
              onClick={() => onToggleSort('lastUpdate')}
              className={`px-4 py-2 text-sm font-medium rounded-md border border-gray-600
                transition-all duration-200 hover:bg-gray-700
                ${sortField === 'lastUpdate' ? 'bg-gray-700 text-blue-400' : 'text-gray-300'}`}
            >
              {t('filter.sort.lastUpdate')}
              <SortIcon field="lastUpdate" />
            </button>
          </div>
        </div>
        
        {/* 搜索输入框 */}
        <div className="relative">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder={t('filter.search')}
            className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-200 
              placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* 已选模型 */}
        {selectedModels.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">{t('filter.selectedModels')}</span>
              <button
                onClick={onClearModels}
                className="text-sm text-gray-400 hover:text-gray-300"
              >
                {t('filter.clearSelection')}
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {selectedModels.map(model => (
                <span
                  key={model}
                  className="inline-flex items-center px-3 py-1 rounded-full text-sm
                    bg-gradient-to-r from-blue-500/20 to-teal-500/20 text-blue-400 border border-blue-500/30"
                >
                  {model}
                  <button
                    onClick={() => onRemoveModel(model)}
                    className="ml-2 text-blue-400 hover:text-blue-300 focus:outline-none"
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* 模型列表 */}
        <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto custom-scrollbar">
          {filteredModels.map(model => (
            <button
              key={model}
              onClick={() => onToggleModel(model)}
              className={`px-3 py-1 rounded-full text-sm font-medium transition-all duration-200
                ${selectedModels.includes(model)
                  ? 'bg-gradient-to-r from-blue-500 to-teal-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
            >
              {model}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
} 