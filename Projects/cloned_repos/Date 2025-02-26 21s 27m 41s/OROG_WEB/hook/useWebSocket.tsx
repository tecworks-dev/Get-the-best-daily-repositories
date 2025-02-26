
import { useEffect, useRef } from "react";
type SubscribeItem = {
    [key: string]: any;
};
// 获取环境变量 socket Url
const href_ws = process.env.NEXT_PUBLIC_BASE_URL_wss || process.env.ENV_HREF_wss
type AllowedStrings = "market_kline" | "market_tx_activity";
type ChannelArrType = AllowedStrings[]
const useWebSocket = (funArr: Function[]) => {
    // socketRef
    const socketRef = useRef<null | WebSocket>(null)
    // timeOut 
    const timeOut = useRef<NodeJS.Timeout | null>(null)
    // socket 是否连接中 0 是没有连接 1 是连接中 2 是关闭
    const isSocketOpen = useRef<0 | 1 | 2>(0)
    // 建立连接函数
    const webSocketInit = (channelArr: ChannelArrType, subscribeData: SubscribeItem[]) => {
        // 建立连接
        const socket = new WebSocket(`${href_ws}/stream`);
        socketRef.current = socket
        // 监听 WebSocket 打开事件
        socket.addEventListener('open', (event) => {
            console.log('WebSocket 连接已打开');
            // 进行订阅，channelArr里面是订阅的频道
            channelArr.forEach((item, index) => {
                sendMessage({
                    id: "UU_!@!##",
                    action: "subscribe",
                    channel: item,
                    data: {
                        ...subscribeData[index]
                    }
                })
            })
            isSocketOpen.current = 1
        });
        // 监听 WebSocket 消息事件
        socket.addEventListener('message', (event) => {
            const data = JSON.parse(event.data);
            receiveMessage(data, channelArr)
            sendPing()
        });

        // 监听 WebSocket 关闭事件
        socket.addEventListener('close', (event) => {
            console.log('WebSocket 连接已关闭:', event);
            isSocketOpen.current = 2
        });

        // 监听 WebSocket 错误事件
        socket.addEventListener('error', (error) => {
            console.log('WebSocket 出现错误:', error);
            isSocketOpen.current = 2
        });
    }
    // 发送 ping 消息 倒计时函数
    const sendPing = () => {
        clearPing && clearPing()
        timeOut.current = setTimeout(() => {
            sendMessage({ action: "ping" })
        }, 15000)
    };
    // 清除 ping 倒计时
    const clearPing = () => {
        clearTimeout(timeOut.current as NodeJS.Timeout)
    }
    // 发送消息的函数
    const sendMessage = (message: any) => {
        if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
            console.log('WebSocket 连接未建立或已关闭');
            return;
        }
        socketRef.current.send(JSON.stringify(message));
        sendPing()
    };
    // 再次进行订阅函数
    const resubscribe = (resubscribeArr: ChannelArrType, subscribeData: SubscribeItem[]) => {
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            resubscribeArr.forEach((item, index) => {
                sendMessage({
                    id: "UU_!@!##",
                    action: "subscribe",
                    channel: item,
                    data: {
                        ...subscribeData[index]
                    }
                })
            })
        }
    }
    // 取消订阅
    const unsubscribe = (unsubscribeArr: ChannelArrType, subscribeData: SubscribeItem[]) => {
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            unsubscribeArr.forEach((item, index) => {
                sendMessage({
                    id: "UU_!@!##",
                    action: "unsubscribe",
                    channel: item,
                    data: {
                        ...subscribeData[index],

                    }
                })
            })
        }
    }
    // 收到消息进行判断
    const receiveMessage = (data: any, channelArr: ChannelArrType) => {
        if (data.action === "pong") return
        const index = channelArr.findIndex((item) => item === data.channel)
        if (index > -1 && data.data) {
            funArr[index](data.data)
        }
    }
    useEffect(() => {
        return () => {
            socketRef.current?.close()
            clearPing()
        }
    }, [])
    return {
        webSocketInit,
        socket: socketRef.current,
        resubscribe,
        sendMessage,
        unsubscribe,
        isSocketOpen: isSocketOpen.current
    }
}

export default useWebSocket