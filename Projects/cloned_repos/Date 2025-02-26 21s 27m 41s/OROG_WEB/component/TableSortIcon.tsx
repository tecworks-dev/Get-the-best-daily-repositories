import { memo } from "react";
import SvgIcon from "./SvgIcon";

interface SortIconProps {
    value: string; // 显示的字段名称
    lightKey: string; // 当前激活排序的字段
    checkedKey: string; // 当前排序状态的字段
    direction: "desc" | "asc" | ""; // 当前排序方向
    classname?: string; // 自定义样式
    onClick?: (direction: "desc" | "asc", key: string) => void; // 点击事件回调
}

const SortIcon: React.FC<SortIconProps> = ({
    value,
    lightKey,
    checkedKey,
    direction,
    classname = "",
    onClick,
}) => {
    // 处理点击事件并切换方向
    const handleSortClick = () => {
        const newDirection =
            lightKey === checkedKey && direction === "desc" ? "asc" : "desc";
        onClick?.(newDirection, lightKey); // 如果 onClick 存在，调用回调
    };

    // 动态设置箭头样式
    const isActive = lightKey === checkedKey;
    const upArrowClass = isActive && direction === "asc" ? "dark:text-white text-4b" : "";
    const downArrowClass = isActive && direction === "desc" ? "dark:text-white text-4b" : "";
    return (
        <div
            className={`flex items-center cursor-pointer justify-center ${classname}`}
            onClick={handleSortClick}
        >
            <div className="mr-1.25">{value}</div>
            <div className="flex flex-col items-center dark:text-4b text-white">
                <SvgIcon stopPropagation={false} value="sort" className={`w-2 ${upArrowClass}`} />
                <SvgIcon stopPropagation={false} value="sort" className={`w-2 -rotate-180 ${downArrowClass}`} />
            </div>
        </div>
    );
};

export default memo(SortIcon);