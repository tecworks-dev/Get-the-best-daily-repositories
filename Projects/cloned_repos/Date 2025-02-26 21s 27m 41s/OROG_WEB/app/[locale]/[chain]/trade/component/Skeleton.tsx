import { Skeleton } from "antd";
import { memo } from "react";

const SkeletonCom = ({ num = 15 }: { num?: number }) => {
    return (
        <div >
            {[...Array(num)].map((_item, index) => {
                return <div className="flex items-center justify-between mb-2" key={index}>
                    <Skeleton.Avatar active className="mr-1 dark:bg-333 rounded-full" size={40} />
                    <div className="flex flex-col justify-between w-full">
                        <Skeleton.Button active className="mb-1 h-5 dark:bg-333" block={true} />
                        <Skeleton.Button active className="h-5 dark:bg-333" block={true} />
                    </div>
                </div>
            })}
        </div>
    )
}

export default memo(SkeletonCom)