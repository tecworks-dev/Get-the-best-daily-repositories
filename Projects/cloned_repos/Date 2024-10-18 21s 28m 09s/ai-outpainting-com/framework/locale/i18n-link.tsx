'use client'
import localeNames from './localeConfig'
import {Link} from "@nextui-org/react";
import {AVAILABLE_LOCALES} from "@/framework/locale/locale";
import {usePathname} from "next/navigation";

export default function I18nLink({params}: { params?: { lang: AVAILABLE_LOCALES } }) {
    // 获取localeNames的keys
    const keys = Object.keys(localeNames)
    // 获取请求路径
    const pathname = usePathname()
    // 删除locale后保留所有路径
    const pathWithoutLocale = pathname.split('/')[2]

    return (
        <div className="grid grid-cols-3 md:grid-cols-8 gap-2">
            {
                keys.map((key) => (
                    <div key={key}>
                        <Link
                            className={params?.lang === key ? "text-primary-200 w-1/7" : "w-1/7 text-gray-500 hover:text-primary-200"}
                            href={`/${key}/${pathWithoutLocale}`}>{
                            // @ts-ignore
                            localeNames[key]
                        }</Link>
                    </div>
                ))
            }
        </div>
    );
}