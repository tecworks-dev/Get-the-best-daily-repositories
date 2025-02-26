import { useEffect } from "react";
const useBrowserTitle = (title: string = 'OROG是最牛逼的交易工具') => {
    useEffect(() => {
        document.title = title;
    }, [title]);
}

export default useBrowserTitle