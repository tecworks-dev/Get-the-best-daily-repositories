import Image from "next/image"
import { mobileHidden } from "@/utils"
import Copy from "./Copy"
import SvgIcon from "./SvgIcon"
import { chainLogo } from "@/app/chainLogo"
import ImageUrl from "./ImageUrl"
import Follow from "./Follow"
import { ChainsType } from "@/i18n/routing"
import { memo } from "react"
const TableTokenInfo = ({ logo, address, symbol, chain, twitter_username, website, telegram, className, follow = false, isFollowShow = false, quote_mint_address = '' }:
    { logo: string, address: string, symbol: string, chain: ChainsType, twitter_username: string, website: string, telegram: string, className?: string, isFollowShow?: boolean, follow?: boolean, quote_mint_address?: string }) => {
    return (
        <div className={` flex items-center ${className}`}>
            {isFollowShow &&
                <Follow className='mr-3.75 w-3.5 min-w-3.5' chain={chain} follow={follow} quote_mint_address={quote_mint_address} />
            }
            <ImageUrl logo={logo} symbol={symbol} className="w-11.25 h-11.25 rounded-full mr-3.75" />
            <div className="">
                <div className="flex items-center mb-0.5">
                    <div className="mr-1.75 dark:text-white text-51 font-semibold text-sm">{symbol}</div>
                    <Image src={chainLogo[chain]} alt="" className="w-3.5" />
                </div>
                <div className="text-13 dark:text-6b text-d3 font-semibold flex items-center">
                    <span className=" mr-2.75">{mobileHidden(address, 4, 4)}</span>
                    <Copy className=" w-3" text={address} />
                    {website && <SvgIcon onClick={() => window.open(website)} className="w-3 ml-1.25" value="website" />}
                    {twitter_username && <SvgIcon className="w-3 ml-1.25" onClick={() => window.open(`https://x.com/${twitter_username}`)} value="x" />}
                    {telegram && <SvgIcon className="w-3 ml-1.25 " onClick={() => window.open(`https://t.me/${telegram}`)} value="telegram" />}
                </div>
            </div>
        </div>
    )
}

export default memo(TableTokenInfo)