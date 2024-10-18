import {PrismaClient} from '@prisma/client';

let prisma: PrismaClient = undefined as any;

if (process.env.NODE_ENV === 'production') {
    prisma = new PrismaClient();
} else {
    if (!prisma) {
        prisma = new PrismaClient({
            log: ['info'],
        });
    }
}

export default prisma;