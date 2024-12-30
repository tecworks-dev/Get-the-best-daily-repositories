import axios from "./index";

export const getUserDetails = async (userId: number) => {
  try {
    const response = await axios.get(`/users/${userId}`);
    return response.data;
  } catch (error) {
    throw new Error("Failed to fetch user details");
  }
};

export const updateUserProfile = async (
  userId: number,
  profileData: object
) => {
  try {
    const response = await axios.put(`/users/${userId}`, profileData);
    return response.data;
  } catch (error) {
    throw new Error("Failed to update profile");
  }
};
