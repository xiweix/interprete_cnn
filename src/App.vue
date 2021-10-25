<template>
  <v-app>
    <v-app-bar app color="primary" dark dense flat>
      <v-toolbar-title class="font-weight-bold subtitle-1">
        Comparative Study of Convolutional Neural Networks through Interactive
        Visualization
      </v-toolbar-title>

      <v-spacer></v-spacer>

      <v-btn
        href="https://github.com/The-Boogeyman/ECS289H_Final"
        target="_blank"
        text
      >
        <span class="mr-2">Github</span>
        <v-icon>fas fa-external-link-alt</v-icon>
      </v-btn>
    </v-app-bar>

    <v-main>
      <MainPage />
    </v-main>
  </v-app>
</template>

<script>
import MainPage from "./components/MainPage";

export default {
  name: "App",

  components: {
    MainPage,
  },

  data: () => ({
    //
  }),
};
</script>
