import { useTranslations } from 'next-intl';
import { toast } from 'react-toastify';

import { ChainsType } from '@/i18n/routing';
import { TokenType } from '@/interface';
import { getTokenHttp } from '@/request';
import { useTokenStore } from '@/store/tokne';
import useConnectWallet from '@/store/wallet/useConnectWallet';

import useMessageSing from './useMessageSign/useMessageSing';

const useLogin = (chain: ChainsType) => {
    const tToast = useTranslations('Toast')
    // 使用消息签名钩子
    const { messageSingFun } = useMessageSing(chain)
    // 获取地址
    const { address } = useConnectWallet(chain)
    // token
    const { token, setLocalToken } = useTokenStore()
    // 进行签名并提交后端获取token存入本地和store
    const singMsgAndSubmit = async () => {
        // 没有地址则代表没有连接钱包
        if (!address) return
        const data = await messageSingFun();
        if (data) {
            try {
                // 提交后端
                const { data: { token } } = await getTokenHttp<TokenType>(address, data.signature)
                // 获取到token后存入本地和store
                setLocalToken(token)
            } catch (e) {
                console.log('getTokenHttp', e);
            }
        }
    }
    // 传入方法，进行检测是否连接钱包，有token，都有执行函数，没有进行提示或者登陆
    const checkExecution = (fun?: Function) => {
        // 判断有没有连接钱包
        if (!address) {
            toast.info(tToast('connect'), {
                progressClassName: 'dark:bg-6f bg-2bc',
            })
            return
        }
        // 判断有没有token，有的执行follow逻辑,,没有进行签名获取token
        if (token) {
            fun && fun()
        } else {
            // 进行签名获取token
            singMsgAndSubmit()
        }
    }

    return {
        singMsgAndSubmit,
        checkExecution
    }
}

export default useLogin