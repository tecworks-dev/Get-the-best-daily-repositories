export const saveToken = (token: string) => {
  localStorage.setItem("authToken", token);
};

export const getToken = () => {
  return localStorage.getItem("authToken");
};

export const clearToken = () => {
  localStorage.removeItem("authToken");
};
