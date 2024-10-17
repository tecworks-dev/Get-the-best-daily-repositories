'use server'
import dictService from "../../../framework/services/DictService";
import {Dict} from "@prisma/client";
import {revalidatePath} from "next/cache";
import {redirect} from "next/navigation";


export async function createDict(state: unknown, dict: Dict) {
    const existDict = await dictService.getByType(dict.type)
    if (existDict != null && !dict.id) {
        return {
            message: "字典类型已存在，不可重复添加"
        }
    }
    if (dict.id) {
        await dictService.update(dict)
    } else {
        await dictService.create(dict)
    }
    revalidatePath("/admin/dict")
    redirect("/admin/dict")
}

export async function deleteDict(dict: Dict) {
    await dictService.delete(dict.id)
    revalidatePath("/admin/dict")
    redirect("/admin/dict")
}
