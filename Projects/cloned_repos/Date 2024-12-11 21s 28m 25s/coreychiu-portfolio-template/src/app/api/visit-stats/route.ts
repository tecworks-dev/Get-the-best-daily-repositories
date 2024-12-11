import { NextResponse } from 'next/server';

const OPENPANEL_API_URL = 'https://api.openpanel.dev';
const OPENPANEL_CLIENT_ID = process.env.NEXT_PUBLIC_OPENPANEL_CLIENT_ID;
const OPENPANEL_SECRET_ID = process.env.OPENPANEL_API_SECRET_ID;
const OPENPANEL_PROJECT_ID = process.env.OPENPANEL_PROJECT_ID;
export async function GET() {
  try {
    // 获取总访问数据
    const response = await fetch(`${OPENPANEL_API_URL}/export/events?projectId=coreychiu&event=screen_view`, {
      headers: {
        'openpanel-client-id': OPENPANEL_CLIENT_ID!,
        'openpanel-client-secret': OPENPANEL_SECRET_ID!,
      },
    });


    // console.log('response: ', response)
    if (!response.ok) {
      throw new Error('Failed to fetch visit stats');
    }


    const data = await response.json();
    // console.log('data: ', data)
    const totalUV = data?.meta?.totalCount;


    // 获取今日访问数据
    // 昨天的 yyyy-MM-dd
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];

    // 今天的 yyyy-MM-dd
    const todayStr = today.toISOString().split('T')[0];
    const todayResponse = await fetch(`${OPENPANEL_API_URL}/export/events?projectId=${OPENPANEL_PROJECT_ID}&event=screen_view&start=${yesterdayStr}&end=${todayStr}`, {
      headers: {
        'openpanel-client-id': OPENPANEL_CLIENT_ID!,
        'openpanel-client-secret': OPENPANEL_SECRET_ID!,
      },
    });

    // console.log('todayResponse: ', todayResponse)
    if (!todayResponse.ok) {
      throw new Error('Failed to fetch visit stats');
    }

    const todayData = await todayResponse.json();
    // console.log('todayData: ', todayData)
    const dailyUV = todayData?.meta?.totalCount;

    return NextResponse.json({
      totalUV,
      dailyUV,
    });
  } catch (error) {
    // console.error('Error fetching visit stats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch visit stats' },
      { status: 500 }
    );
  }
}