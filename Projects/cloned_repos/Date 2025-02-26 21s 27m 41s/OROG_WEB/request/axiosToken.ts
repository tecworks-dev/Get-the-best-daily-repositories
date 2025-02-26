import { useTokenStore } from "@/store/tokne";
import axios, {
  AxiosRequestConfig,
  AxiosRequestHeaders,
  AxiosResponse,
} from "axios";
const href = process.env.NEXT_PUBLIC_BASE_URL || process.env.ENV_HREF;
const $axiosToken = axios.create({
  baseURL: href, // 设置你的API基础URL
  timeout: 30000, // 设置请求超时时间
  headers: {
    "Content-Type": "application/json",
  },
});
// 添加请求拦截器
$axiosToken.interceptors.request.use(
  (config: AxiosRequestConfig & { headers: AxiosRequestHeaders }) => {
    // 在请求发送之前可以做一些处理，例如添加请求头等
    config.headers.Authorization = "Bearer " + useTokenStore.getState().token; // 你可以添加身份验证头部
    return config;
  },
  (error: any) => {
    // 对请求错误做些什么
    return Promise.reject(error);
  }
);

// 响应拦截器
$axiosToken.interceptors.response.use(
  (response: AxiosResponse) => {
    // 在响应数据之前可以做一些处理
    if (response.data.code === 0) {
      return response;
    } else {
      return Promise.reject(response);
    }
  },
  (error: any) => {
    // 处理错误响应
    return Promise.reject(error);
  }
);
export default $axiosToken;
