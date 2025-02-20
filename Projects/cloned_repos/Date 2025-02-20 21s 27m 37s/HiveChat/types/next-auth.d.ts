import NextAuth from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      name: string;
      email: string;
      isAdmin?: boolean; // 扩展 isAdmin 字段
    }
  }

  interface User {
    id: string;
    name: string;
    email: string;
    isAdmin?: boolean; // 扩展 isAdmin 字段
  }

  interface JWT {
    id: string;
    name?: string;
    email?: string;
    isAdmin?: boolean; // 扩展 isAdmin 字段
  }
}