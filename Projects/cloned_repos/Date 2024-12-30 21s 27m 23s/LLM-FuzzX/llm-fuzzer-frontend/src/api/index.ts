import axios from "axios";

const instance = axios.create({
  baseURL:
    process.env.NODE_ENV === "development"
      ? "/api"
      : "http://localhost:10003/api",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

export default instance;
