import { Module } from "vuex";

interface UserState {
  username: string;
  userId: number | null;
  preferences: {
    theme: string;
  };
}

const user: Module<UserState, any> = {
  namespaced: true,
  state: {
    username: "",
    userId: null,
    preferences: {
      theme: "light",
    },
  },
  mutations: {
    setUsername(state, username: string) {
      state.username = username;
    },
    setUserId(state, userId: number) {
      state.userId = userId;
    },
    setTheme(state, theme: string) {
      state.preferences.theme = theme;
    },
  },
  actions: {
    updateUserTheme({ commit }, theme: string) {
      commit("setTheme", theme);
    },
  },
};

export default user;
