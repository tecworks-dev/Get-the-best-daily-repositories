import { Tooltip } from "antd"
import { TooltipPlacement } from "antd/es/tooltip";
import { useTranslations } from "next-intl";
import { memo } from "react";
interface TableTooltipProps {
    children: React.ReactNode;
    title?: string;
    placement?: TooltipPlacement
}
const TableTooltip = ({ children, title, placement = 'top' }: TableTooltipProps) => {
    const t = useTranslations('TableTooltip')
    return (
        <Tooltip arrow={false} placement={placement} trigger='hover' title={title || t('top10Holders30')}>
            <div>{children}</div>
        </Tooltip>
    )
}

export default memo(TableTooltip)