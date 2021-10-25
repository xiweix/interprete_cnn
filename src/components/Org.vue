<template>
  <v-container fluid v-if="OrgImg !== null">
    <p>Original (label: {{ label }})</p>
    <v-img :src="OrgImg" width="150" contain></v-img>
  </v-container>
</template>

<script>
export default {
  name: "Org",

  data: () => ({
    OrgImg: null,
    label: null,
  }),
  mounted() {
    this.$options.sockets.onmessage = (res) => {
      res = res.data;
      if (res.indexOf("sample_img***") !== -1) {
        res = res.split("***");
        this.OrgImg = "data:image/png;base64," + res[1];
        this.label = res[2];
      }
    };
  },
};
</script>
