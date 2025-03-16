import { NextResponse } from 'next/server';
import { getAuthConfigOnServer, getFamilyDataOnServer } from '@/utils/config';

// 验证token是否有效
async function verifyToken(token: string | null): Promise<boolean> {
  if (!token) return false;
  
  try {
    // 使用Buffer解码token
    const tokenData = Buffer.from(token, 'base64').toString();
    const parsedToken = JSON.parse(tokenData);
    const expirationTime = parsedToken.exp;
    const currentTime = Date.now();
    
    // 检查token是否过期（24小时）
    return currentTime < expirationTime;
  } catch (error) {
    console.error('Token验证错误:', error);
    return false;
  }
}

export async function GET(request: Request) {
  try {
    // 从URL中获取查询参数
    const url = new URL(request.url);
    const configType = url.searchParams.get('type');
    
    // 获取认证配置
    const authConfig = await getAuthConfigOnServer();
    
    // 如果需要认证，检查Authorization header
    if (authConfig.requireAuth) {
      const authHeader = request.headers.get('Authorization');
      const token = authHeader ? authHeader.replace('Bearer ', '') : null;
      
      // 验证token是否有效
      const isValidToken = await verifyToken(token);
      
      if (!isValidToken) {
        return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
      }
    }
    
    // 根据不同的配置类型返回不同的数据
    if (configType === 'auth') {
      // 只返回公共配置信息，不包含敏感信息
      const publicConfig = {
        familyName: authConfig.familyName,
        requireAuth: authConfig.requireAuth
      };
      return NextResponse.json(publicConfig);
    } else if (configType === 'family') {
      const familyData = await getFamilyDataOnServer();
      return NextResponse.json(familyData);
    } else {
      // 如果没有指定类型或类型无效，返回错误
      return NextResponse.json({ error: 'Invalid config type' }, { status: 400 });
    }
  } catch (error) {
    console.error('获取配置时出错:', error);
    return NextResponse.json({ error: 'Failed to load config' }, { status: 500 });
  }
} 