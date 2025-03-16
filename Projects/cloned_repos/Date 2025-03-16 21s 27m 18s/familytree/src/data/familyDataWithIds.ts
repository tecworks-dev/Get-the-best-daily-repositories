'use client';

import { useState, useEffect } from 'react';
import { FamilyData } from '../types/family';
import { getFamilyData } from '../utils/config';

// 从环境变量中创建的默认数据（在无法通过API获取数据时使用）
export const familyDataWithIds: FamilyData = getFamilyData();

// 用于在客户端获取家族数据的钩子
export function useFamilyData(): { 
  data: FamilyData; 
  loading: boolean; 
  error: string | null 
} {
  const [data, setData] = useState<FamilyData>(familyDataWithIds);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        
        // 获取认证令牌
        const token = localStorage.getItem('auth_token');
        
        // 准备请求头，如果有token则添加到Authorization头
        const headers: HeadersInit = {
          'Content-Type': 'application/json'
        };
        
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }
        
        // 发送带认证令牌的请求
        const response = await fetch('/api/config?type=family', {
          headers
        });
        
        if (response.status === 401) {
          // 未授权，清除token并提示用户
          localStorage.removeItem('auth_token');
          localStorage.removeItem('auth_time');
          throw new Error('认证已过期，请重新登录');
        }
        
        if (!response.ok) {
          throw new Error(`API返回错误状态: ${response.status}`);
        }
        
        const fetchedData = await response.json();
        setData(fetchedData);
        setError(null);
      } catch (err) {
        console.error('获取家族数据失败:', err);
        setError('加载家族数据失败，使用默认数据');
        // 出错时保留默认数据
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  return { data, loading, error };
} 