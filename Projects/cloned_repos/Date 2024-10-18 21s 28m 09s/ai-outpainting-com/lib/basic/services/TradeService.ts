import {Trade} from "@prisma/client";
import prisma from "@/config/prisma";
import {isNumber} from "radash";

const services = {
    async list(data: Trade) {
        return prisma.trade.findMany({
            where: {...data}
        })
    },
    async create(data: Trade) {
        return prisma.trade.create({
            data
        })
    },
    async updateById(data: Trade) {
        const {id} = data
        return prisma.trade.update({
            data,
            where: {
                id
            }
        })
    },
    async deleteById(data: Trade | number) {
        const id = isNumber(data) ? data : data.id
        return prisma.trade.delete({
            where: {
                id
            }
        })
    }
}

export default services