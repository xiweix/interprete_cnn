<template>
  <v-container fluid v-if="model1Prd !== null">
    <v-row no-gutters>
      <v-col cols="1">
        <br />
        Model1
        <br />
        (Epoch: {{ epoch }})
        <br />
        (Prediction: {{ model1Prd }})
      </v-col>
      <v-col v-for="act1 in model1Activations" :key="act1.id">
        <div>{{ act1.name }}</div>
        <v-img :src="act1.img" width="100" contain></v-img>
      </v-col>
    </v-row>
    <v-row no-gutters>
      <v-col cols="1">
        <br />
        Model2
        <br />
        (Epoch: {{ epoch }})
        <br />
        (Prediction: {{ model2Prd }})
      </v-col>
      <v-col v-for="act2 in model2Activations" :key="act2.id">
        <div>{{ act2.name }}</div>
        <v-img :src="act2.img" width="100" contain></v-img>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: "activationAnnotation",

  data: () => ({
    model1Prd: null,
    model2Prd: null,
    epoch: null,
    model1Activations: [],
    model2Activations: [],
  }),
  mounted() {
    this.$options.sockets.onmessage = (res) => {
      res = res.data;
      if (res.indexOf("model1AnnotationActivations***") !== -1) {
        this.model1Activations = [];
        res = res.split("***");
        const imgLength = parseInt(res[1]);
        this.model1Prd = res[2];
        this.epoch = res[3];
        let i = 0;
        for (i = 0; i < imgLength; i++) {
          this.model1Activations.push({
            id: i,
            name: res[4 + 2 * i],
            img: "data:image/png;base64," + res[5 + 2 * i],
          });
        }
      } else if (res.indexOf("model2AnnotationActivations***") !== -1) {
        this.model2Activations = [];
        res = res.split("***");
        const imgLength = parseInt(res[1]);
        this.model2Prd = res[2];
        let i = 0;
        for (i = 0; i < imgLength; i++) {
          this.model2Activations.push({
            id: i,
            name: res[4 + 2 * i],
            img: "data:image/png;base64," + res[5 + 2 * i],
          });
        }
      } else if (res.indexOf("clearActivations2***") !== -1) {
        this.model1Prd = null;
        this.model2Prd = null;
        this.epoch = null;
        this.model1Activations = [];
        this.model2Activations = [];
      }
    };
  },
};
</script>
