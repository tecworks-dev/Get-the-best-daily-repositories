import { ConfigProvider, Skeleton, Table } from "antd";
import "@/app/antd-table.scss"
import { useRouter } from "@/i18n/routing";
import Empty from "./Empty";
import { memo } from "react";
const AppTable = ({ columns, data, loading = false, keys, mh, skeletonMh, skeletonClassname,
    isCilCk = true, arr = 40, isScroll = true }:
    { columns: any, data: any, loading: boolean, keys: string, mh: string, skeletonMh: string, skeletonClassname?: string, isCilCk?: boolean, arr?: number, isScroll?: boolean }) => {
    const router = useRouter();
    return (
        <ConfigProvider
            theme={{
                components: {
                    Table: {
                        colorBgContainer: "var(--bg-color)", // 组件的容器背景色
                        headerBg: "var(--bg-color)", // 表头背景色
                        headerBorderRadius: 9, // 表头圆角
                        headerColor: "var(--antd-table-header-color)", // 表头文字颜色
                        fontSize: 13, // 头部字体大小
                        colorText: "var(--text-color)", // 文本颜色
                        cellPaddingBlock: 0,
                        cellPaddingInline: 0
                    },
                    Skeleton: {
                        paragraphLiHeight: 50
                    }
                },
            }}
        >
            <Table
                onRow={(record) => {
                    return {
                        onClick: () => {
                            isCilCk && router.push(`/trade/${record.address}`);
                        }, // 点击行的回调
                    };

                }}
                className={` w-full  overflow-hidden whitespace-nowrap`}
                scroll={isScroll ? { y: `calc(100vh - ${mh})` } : undefined}
                locale={{
                    emptyText: loading ? '' : <Empty />
                }}
                rowKey={(record) => {
                    return record[keys]
                }} rowClassName={isCilCk ? 'cursor-pointer' : ''} columns={columns} dataSource={data} pagination={false} rowHoverable={false} />
            {loading &&
                <div className={`w-full  overflow-hidden ${skeletonClassname}`} style={{ height: `calc(100vh - ${skeletonMh})` }}>
                    {Array(arr)
                        .fill(null)
                        .map((_, index) => {
                            return (
                                <Skeleton.Node
                                    key={index} // 添加唯一的 key
                                    active
                                    className="block dark:bg-black1F w-full h-12 mb-3"
                                />
                            );
                        })}
                </div>}
        </ConfigProvider>
    );
};

export default memo(AppTable);