import { MyResponseType } from "@/interface";
import $axios from "./axios";
import $axiosToken from "./axiosToken";
// 发送 GET 请求的函数示例
export const getRequest = async <T>(
  url: string,
  params?: any
): Promise<MyResponseType<T>> => {
  try {
    const { data } = await $axios.get(url, { params });
    return data;
  } catch (e: any) {
    throw e;
  }
};

// 发送 POST 请求的函数示例
export const postRequest = async <T>(
  url: string,
  params?: any
): Promise<MyResponseType<T>> => {
  try {
    const { data } = await $axios.post(url, params);
    return data;
  } catch (e: any) {
    throw e;
  }
};
// 发送 GET 请求的函数示例 携带token
export const getRequestToken = async <T>(
  url: string,
  params?: any
): Promise<MyResponseType<T>> => {
  try {
    const { data } = await $axiosToken.get(url, { params });
    return data;
  } catch (e: any) {
    throw e;
  }
};

// 发送 POST 请求的函数示例 携带token
export const postRequestToken = async <T>(
  url: string,
  params?: any
): Promise<MyResponseType<T>> => {
  try {
    const { data } = await $axiosToken.post(url, params);
    return data;
  } catch (e: any) {
    throw e;
  }
};
