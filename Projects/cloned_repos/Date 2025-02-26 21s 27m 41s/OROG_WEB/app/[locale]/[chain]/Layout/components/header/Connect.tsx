import {
  memo,
  useEffect,
  useState,
} from 'react';

// 导入必要的组件和工具
import {
  Dropdown,
  DropdownProps,
} from 'antd';
import { useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';
import Image from 'next/image';

import useLogin from '@/hook/useLogin';
import {
  ChainsType,
  useRouter,
} from '@/i18n/routing';
import bitgit from '@/public/bitgit.png';
import okx from '@/public/okx.png';
import phantom from '@/public/phantom.png';
import snake from '@/public/snake.jpg';
import { useTokenStore } from '@/store/tokne';
import useConnectWallet from '@/store/wallet/useConnectWallet';
import { useConnectWalletSol } from '@/store/wallet/useConnectWalletInt';
import { mobileHidden } from '@/utils';

const Copy = dynamic(() => import('@/component/Copy'));
const SvgIcon = dynamic(() => import('@/component/SvgIcon'));
// 定义钱包列表，包含钱包的关键信息和图标
const walletList = [
    {
        key: 'OkxWallet',
        label: 'okx wallet',
        img: okx
    },
    {
        key: "Phantom",
        label: 'Phantom',
        img: phantom
    },
    {
        key: 'BitgetWallet',
        label: 'Bitget Wallet',
        img: bitgit
    },
];

/**
 * Connect组件用于用户连接到Solana链上的钱包
 * @param {Object} props
 * @param {ChainsType} props.chain - 当前的链类型
 */
const Connect = ({ chain }: { chain: ChainsType }) => {
    // 翻译
    const t = useTranslations('Header')
    // 使用路由器
    const router = useRouter()
    // 使用Token状态管理
    const { token, removeLocalToken } = useTokenStore()
    // 使用钱包状态管理
    const { address,
        connectWalletStore,
        disconnectWalletStore,
    } = useConnectWallet(chain)
    // 使用useConnectWalletSol初始化连接地址，因为浏览器注入钱包对象比较慢，所以需要初始化连接地址
    useConnectWalletSol(chain, removeLocalToken)
    // 签名和获取token hooks
    const { singMsgAndSubmit } = useLogin(chain)
    // 状态管理，用于控制Dropdown的打开状态
    const [open, setOpen] = useState(false);
    // 连接钱包函数
    const connectWallet = (key: string) => {
        connectWalletStore(key, removeLocalToken)
        setOpen(false)
    }
    // 断开钱包函数
    const disconnectWallet = () => {
        disconnectWalletStore();
        setOpen(false);
        removeLocalToken()
    }
    /**
     * 处理Dropdown打开状态变化
     * @param {boolean} nextOpen - 下一个打开状态
     * @param {Object} info - 相关信息
     */
    const handleOpenChange: DropdownProps['onOpenChange'] = (nextOpen, info) => {
        if (info.source === 'trigger' || nextOpen) {
            setOpen(nextOpen);
        }
    };
    // 定义钱包操作列表
    const wallets = [
        // { title: t('myWallet'), fun: () => { router.push(`/mywallet/${address}`); setOpen(false) } },
        // { title: t('refer'), fun: () => { router.push(`/referral`); setOpen(false) } },
        { title: t('disconnect'), fun: () => { disconnectWallet() } },
    ]
    // 当Solana链的提供者变化时，可能需要重新认证
    useEffect(() => {
        if (address && !token) {
            singMsgAndSubmit()
        }
    }, [address, token])
    // 渲染连接钱包的组件
    return (
        <div className="ml-6.25 cursor-pointer custom830:ml-2">
            <Dropdown
                trigger={['click']}
                open={open}
                onOpenChange={handleOpenChange}
                dropdownRender={() => (
                    <div className=" mt-2">
                        {address ? <div className=" cursor-pointer border dark:bg-gray4b bg-white dark:border-2c border-d9 rounded-9 dark:text-e7">
                            {wallets.map(({ title, fun }) => (
                                <div key={title} className="border-b-2 dark:border-2c border-d3  last:border-none flex items-center justify-center text-13 font-medium h-7 " onClick={fun}>{title}</div>
                            ))}
                        </div> : <div className="w-87.25 relative box-border dark:text-white text-black dark:bg-black bg-white dark:border-2c border-zinc-400 border-2 rounded-2.5xl  pt-11.25 pb-6.25 px-5.75">
                            <SvgIcon value="close" onClick={() => setOpen(false)} className="absolute  top-3.25 left-4  cursor-pointer w-3.25 dark:text-white" />
                            <p className="text-center text-xl font-semibold mb-4.75">{t('connectWalletTitle')}</p>
                            <div>
                                {walletList.map((item, index) => (
                                    <div key={index} onClick={() => { connectWallet(item.key) }}
                                        className="cursor-pointer mb-3 last:mb-0 box-border flex items-center justify-between border-2 rounded-4xl
                                    dark:bg-black0E bg-slate-100 dark:hover:bg-zinc-800  hover:bg-slate-200  dark:border-2d  border-zinc-400 pr-6.25 pl-0.75 py-0.75  ">
                                        <Image src={item.img} alt='' className="w-12.5" />
                                        <div className="font-semibold text-1.5xl">{item.label}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                        }
                    </div>
                )}
            >
                {address ? <div className="box-border flex items-center rounded-4xl dark:bg-black bg-white dark:text-83 text-black dark:border-black border-d3  border py-0.25 pl-0.25 pr-3.065">
                    <Image className="w-9.5 rounded-full" src={snake} alt="" />
                    <p className="mx-1.5 text-13 font-semibold">{mobileHidden(address, 4, 3)}</p>
                    <Copy text={address} className="mr-2.25 w-3.25" />
                    <SvgIcon value="arrowsBottom" className={`w-3.25 text-7a transform-[0.3s] ${open ? 'rotate-180' : 'rotate-0'}`} />
                </div> :
                    <div className="w-40.25 h-10 rounded-4xl bg-white box-border border dark:border-white border-black text-black text-xl font-bold flex items-center  justify-center">
                        {t('connect')}
                    </div>}
            </Dropdown>

        </div>
    )
}

export default memo(Connect)

