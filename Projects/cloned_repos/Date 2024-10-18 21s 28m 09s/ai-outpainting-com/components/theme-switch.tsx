"use client";
import type {ComponentType, FC} from 'react';
import {SwitchProps, useSwitch} from "@nextui-org/switch";
import {useTheme} from "next-themes";
import {useIsSSR} from "@react-aria/ssr";

export interface ThemeSwitchProps {
    className?: string;
    classNames?: SwitchProps["classNames"];
}

export const ThemeSwitch: FC<ThemeSwitchProps> = ({className, classNames,}) => {
    const {theme, setTheme} = useTheme();
    const isSSR = useIsSSR();

    const onChange = () => {
        theme === "light" ? setTheme("dark") : setTheme("light");
    };

    const {
        Component: SwitchComponent, // 更改名称以避免与内置的 'Component' 关键字冲突，并明确其为一个组件
        slots,
        isSelected,
        getBaseProps,
        getInputProps,
        getWrapperProps,
    } = useSwitch({
        isSelected: theme === "light" || isSSR,
        "aria-label": `Switch to ${theme === "light" || isSSR ? "dark" : "light"} mode`,
        onChange,
    });

    const Switch: ComponentType<any> = SwitchComponent as ComponentType<any>; // 强制转换为正确的类型

    return (
        <div></div>
        /*<Switch
            {...getBaseProps({
                className: clsx(
                    "px-px transition-opacity hover:opacity-80 cursor-pointer",
                    className,
                    classNames?.base
                ),
            })}
        >
            <VisuallyHidden>
                <input {...getInputProps()} />
            </VisuallyHidden>
            <div
                {...getWrapperProps()}
                className={slots.wrapper({
                    class: clsx(
                        [
                            "w-auto h-auto",
                            "bg-transparent",
                            "rounded-lg",
                            "flex items-center justify-center",
                            "group-data-[selected=true]:bg-transparent",
                            "!text-default-500",
                            "pt-px",
                            "px-0",
                            "mx-0",
                        ],
                        classNames?.wrapper
                    ),
                })}
            >
             {!isSelected || isSSR ? <SunFilledIcon size={22} /> : <MoonFilledIcon size={22} />}
            </div>
        </Switch>*/
    );
};
