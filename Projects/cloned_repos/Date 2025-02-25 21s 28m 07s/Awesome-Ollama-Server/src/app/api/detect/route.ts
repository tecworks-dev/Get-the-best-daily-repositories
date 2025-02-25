import { NextResponse } from 'next/server';
import { checkService, measureTPS } from '@/lib/detect';

export const maxDuration = 50; // 设置最大执行时间为 300 秒

export async function POST(request: Request) {
  let url = '';
  try {
    const { url: requestUrl } = await request.json();
    url = requestUrl;
    
    if (!url) {
      return NextResponse.json({ error: 'URL is required' }, { status: 400 });
    }

    console.log('检测服务:', url);

    // 检查服务并获取可用模型
    const models = await checkService(url);
    
    // 如果服务不可用，返回空结果
    if (!models) {
      return NextResponse.json({
        server: url,
        models: [],
        tps: 0,
        lastUpdate: new Date().toISOString(),
        status: 'error'
      });
    }

    // 如果有可用模型，测试性能
    let tps = 0;
    if (models.length > 0) {
      try {
        tps = await measureTPS(url, models[0]);
      } catch (error) {
        console.error('性能测试失败:', error);
      }
    }

    // 返回结果
    return NextResponse.json({
      server: url,
      models: models.map(model => model.name),
      tps,
      lastUpdate: new Date().toISOString(),
      status: 'success'
    });

  } catch (error) {
    console.error('检测出错:', error);
    return NextResponse.json({
      server: url,
      models: [],
      tps: 0,
      lastUpdate: new Date().toISOString(),
      status: 'error'
    });
  }
}